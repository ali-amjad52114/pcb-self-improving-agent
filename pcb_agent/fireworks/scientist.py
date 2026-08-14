"""The Fireworks ML scientist: diagnose a failure, then choose an experiment.

Split into two calls on purpose rather than one.

* `diagnose` is schema-constrained JSON. Its `failure_signature.summary` is the
  string that gets embedded and used as the Atlas Vector Search key, so it has
  to be stable and well-formed, which constrained decoding gives us.
* `propose_experiment` is tool calling with `tool_choice="required"`. The model
  cannot answer in prose; it must pick one executable action from the
  allowlist. That is the difference between an agent and a chat log.

Retrieved memory is injected in two ways, and the second one is the one that
matters. Lessons appear in the prompt, and lessons marked `avoid` for the
current failure mode are removed from the tool allowlist entirely. Prompt text
can be ignored by a model; a missing tool cannot be called.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Sequence

from .client import FireworksClient
from .models import Role
from .schemas import DIAGNOSIS_SCHEMA, PROPOSAL_SCHEMA
from .structured import complete_json
from .tools import (
    TOOL_SPECS,
    ToolCall,
    force_tool,
    parse_tool_calls,
    tool_definitions,
)

log = logging.getLogger("pcb_agent.fireworks.scientist")

SCIENTIST_SYSTEM = """\
You are the ML scientist for an autonomous PCB defect classification agent.

Your job is to read evaluation results, work out why the classifier is failing, \
and choose exactly one experiment that is most likely to raise macro F1.

Rules:
- Ground every claim in the numbers you are given. Cite specific per-class \
metrics or confusion-matrix cells. Do not speculate about data you cannot see.
- Prior lessons are real evidence from earlier runs on related problems. When a \
lesson matches the current failure, prefer it over a generic first guess, and \
say which lesson you used.
- A lesson marked AVOID means that intervention was tried on this failure mode \
and did not help. Do not repeat it. Choose something else.
- Change one thing at a time. A compound change makes the outcome \
uninterpretable and pollutes the memory the next run will read.
- Prefer the cheapest intervention that could plausibly explain the whole gap. \
Threshold moves cost no training time; architecture swaps cost the most.
- Be honest about confidence. Low confidence triggers an independent second \
opinion, which is cheaper than a wasted training run.
"""

PROPOSER_SYSTEM = """\
You are the ML scientist choosing the next experiment for a PCB defect classifier.

You must call exactly one tool. Do not reply with prose. The tool you call is \
executed for real: it mutates the training config and triggers a retrain, so \
the arguments must be concrete and immediately runnable.

Pick the single intervention best supported by the diagnosis and by prior \
lessons. Give a `reason` that names the evidence you acted on.
"""


def diagnose(
    client: FireworksClient,
    state: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]] | None = None,
    *,
    vision_report: Mapping[str, Any] | None = None,
    role: Role = Role.SCIENTIST,
) -> dict[str, Any]:
    """Frozen interface: `diagnose(state, memories) -> dict`.

    `state` is the LangGraph AgentState as a plain dict. Expected keys:
    `dataset_summary`, `current_config`, `current_metrics`, `confusion_matrix`,
    `experiment_history`. All are optional; missing ones are simply omitted.
    """
    memories = list(memories or [])
    user = _render_diagnosis_prompt(state, memories, vision_report)

    result = complete_json(
        client,
        name="Diagnosis",
        schema=DIAGNOSIS_SCHEMA,
        messages=[
            {"role": "system", "content": SCIENTIST_SYSTEM},
            {"role": "user", "content": user},
        ],
        role=role,
        purpose="diagnose",
    )

    # The model is asked to cite the lessons it used; make sure the ids it
    # names actually exist, so a hallucinated citation never reaches MongoDB.
    known_ids = {str(m.get("lesson_id")) for m in memories if m.get("lesson_id")}
    cited = [lid for lid in result.get("memory_used", []) if str(lid) in known_ids]
    dropped = len(result.get("memory_used", [])) - len(cited)
    if dropped:
        log.warning("dropped %d hallucinated lesson citation(s)", dropped)
    result["memory_used"] = cited
    result["cold_start"] = not memories
    return result


def propose_experiment(
    client: FireworksClient,
    state: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    *,
    memories: Sequence[Mapping[str, Any]] | None = None,
    allowed_actions: Sequence[str] | None = None,
    force: str | None = None,
    role: Role = Role.SCIENTIST,
) -> dict[str, Any]:
    """Frozen interface: `propose_experiment(state, diagnosis) -> dict`.

    Returns a dict matching `PROPOSAL_SCHEMA`, ready for `apply_experiment`.
    """
    memories = list(memories or [])
    allowed = list(allowed_actions) if allowed_actions else list(TOOL_SPECS)
    tools = tool_definitions(allowed)
    user = _render_proposal_prompt(state, diagnosis, memories, allowed)

    payload = client.chat(
        [
            {"role": "system", "content": PROPOSER_SYSTEM},
            {"role": "user", "content": user},
        ],
        role=role,
        tools=tools,
        tool_choice=force_tool(force) if force else "required",
        parallel_tool_calls=False,  # one change at a time, by design
        temperature=0.1,            # docs: 0.0-0.3 for reliable tool selection
        purpose="propose_experiment",
    )

    calls = parse_tool_calls(payload, allowed=allowed)
    if len(calls) > 1:
        log.warning("model proposed %d actions; keeping the first (%s)",
                    len(calls), calls[0].name)
    call = calls[0]

    proposal = call.as_proposal()
    proposal.update(
        {
            "expected_effect": _expected_effect(call, diagnosis),
            "confidence": float(diagnosis.get("confidence", 0.5)),
            "second_opinion_requested": _wants_second_opinion(state, diagnosis),
            "tool_call_id": call.call_id,
            "allowed_actions": allowed,
        }
    )
    return proposal


def choose_experiment(
    client: FireworksClient,
    state: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convenience for the LangGraph node: diagnose then propose in one call."""
    diagnosis = diagnose(client, state, memories)
    proposal = propose_experiment(client, state, diagnosis, memories=memories, **kwargs)
    return diagnosis, proposal


# ------------------------------------------------------------------- prompts


def _render_diagnosis_prompt(
    state: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
    vision_report: Mapping[str, Any] | None,
) -> str:
    sections = [
        _section("DATASET", state.get("dataset_summary")),
        _section("CURRENT CONFIG", state.get("current_config")),
        _section("CURRENT METRICS", state.get("current_metrics")),
        _section("CONFUSION MATRIX", state.get("confusion_matrix")),
        _render_history(state.get("experiment_history")),
        _render_memories(memories),
    ]
    if vision_report:
        sections.append(_section("VISUAL INSPECTION OF MISCLASSIFIED CROPS", vision_report))

    sections.append(
        "TASK\nDiagnose the primary reason macro F1 is where it is. Write a "
        "failure_signature whose summary would let a future run retrieve this "
        "experience by semantic similarity: describe the failure, not this "
        "run's identifiers."
    )
    return "\n\n".join(s for s in sections if s)


def _render_proposal_prompt(
    state: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
    allowed: Sequence[str],
) -> str:
    sections = [
        _section("DIAGNOSIS", diagnosis),
        _section("CURRENT CONFIG", state.get("current_config")),
        _section("CURRENT METRICS", state.get("current_metrics")),
        _render_history(state.get("experiment_history")),
        _render_memories(memories),
    ]

    excluded = sorted(set(TOOL_SPECS) - set(allowed))
    if excluded:
        sections.append(
            "WITHDRAWN ACTIONS\nThese were removed from your toolset because prior "
            "lessons show they did not help this failure mode: "
            + ", ".join(excluded)
        )

    sections.append("TASK\nCall exactly one tool with concrete arguments.")
    return "\n\n".join(s for s in sections if s)


def _section(title: str, payload: Any) -> str:
    if payload in (None, {}, [], ""):
        return ""
    body = payload if isinstance(payload, str) else json.dumps(payload, indent=2,
                                                               default=str)
    return f"{title}\n{body}"


def _render_history(history: Any, limit: int = 8) -> str:
    """Compact history. Full experiment records would dominate the context
    window by iteration 4, and the model only needs action plus outcome."""
    if not history:
        return "EXPERIMENT HISTORY\n(none, this is the first experiment of the run)"
    rows = list(history)[-limit:]
    lines = []
    for index, item in enumerate(rows, start=len(history) - len(rows) + 1):
        action = item.get("action") or item.get("next_action") or "?"
        before = _f(item.get("before_macro_f1"))
        after = _f(item.get("after_macro_f1"))
        delta = item.get("delta")
        delta_s = f"{delta:+.3f}" if isinstance(delta, (int, float)) else "?"
        lines.append(f"  {index}. {action}: {before} -> {after} ({delta_s})")
    return "EXPERIMENT HISTORY (this run)\n" + "\n".join(lines)


def _render_memories(memories: Sequence[Mapping[str, Any]]) -> str:
    if not memories:
        return ("RETRIEVED PRIOR LESSONS\n(none: cold start. No comparable "
                "experience exists yet, so reason from the metrics alone.)")
    lines = []
    for memory in memories:
        marker = "AVOID" if memory.get("avoid") else "USE"
        score = memory.get("score")
        score_s = f" similarity={score:.3f}" if isinstance(score, (int, float)) else ""
        lines.append(
            f"- [{marker}] {memory.get('lesson_id', '?')}{score_s}\n"
            f"    failure:      {memory.get('failure_summary', '')}\n"
            f"    intervention: {memory.get('intervention', '')}\n"
            f"    result:       {memory.get('result', '')}\n"
            f"    lesson:       {memory.get('lesson', '')}"
        )
    return "RETRIEVED PRIOR LESSONS (from MongoDB Vector Search)\n" + "\n".join(lines)


def _f(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "?"


def _expected_effect(call: ToolCall, diagnosis: Mapping[str, Any]) -> str:
    signature = diagnosis.get("failure_signature") or {}
    classes = signature.get("affected_classes") or []
    target = f" on {', '.join(classes[:3])}" if classes else ""
    return f"{call.name} should raise macro F1{target}: {call.reason}"


def _wants_second_opinion(state: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> bool:
    """The brief's escalation policy, evaluated here so the LangGraph node is
    a simple `if proposal['second_opinion_requested']`.

    Trigger on low confidence, two consecutive failed experiments, or a model
    family change.
    """
    if float(diagnosis.get("confidence", 1.0)) < 0.55:
        return True

    history = list(state.get("experiment_history") or [])
    recent = history[-2:]
    if len(recent) == 2 and all(
        isinstance(item.get("delta"), (int, float)) and item["delta"] <= 0
        for item in recent
    ):
        return True
    return False


def validate_proposal(proposal: Mapping[str, Any]) -> list[str]:
    """Schema-check a proposal before it is executed. Used by the orchestrator
    when a proposal arrives from somewhere other than `propose_experiment`."""
    from .structured import validate as _validate

    errors = _validate(
        {k: v for k, v in proposal.items() if k in PROPOSAL_SCHEMA["properties"]},
        PROPOSAL_SCHEMA,
    )
    action = proposal.get("next_action")
    if action in TOOL_SPECS:
        params = dict(proposal.get("parameters") or {})
        params.setdefault("reason", proposal.get("rationale") or "unspecified")
        errors.extend(
            f"parameters: {e}"
            for e in _validate(params, TOOL_SPECS[action]["function"]["parameters"])
        )
    elif action is not None:
        errors.append(f"next_action {action!r} is not an allowed tool")
    return errors


__all__ = [
    "diagnose",
    "propose_experiment",
    "choose_experiment",
    "validate_proposal",
    "SCIENTIST_SYSTEM",
    "PROPOSER_SYSTEM",
]
