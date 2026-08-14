"""The Fireworks critic: judge an experiment, then write the reusable lesson.

The critic is the component that makes the whole "no cold start" claim work.
Every lesson it writes is what a future run retrieves, so the quality bar is
different from the scientist's: a lesson that mentions run ids, experiment
numbers, or this dataset's exact class names is useless to the next run.

Run at temperature 0 by default. The critic is a judge, and judges should not
be creative.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .client import FireworksClient
from .models import Role
from .schemas import CRITIQUE_SCHEMA
from .structured import complete_json

log = logging.getLogger("pcb_agent.fireworks.critic")

CRITIC_SYSTEM = """\
You are the critic for an autonomous PCB defect classification agent.

You are given the metrics before an experiment, the intervention that was \
applied, and the metrics after. Decide whether the scientist's hypothesis was \
correct, explain why the intervention helped or failed, and write one reusable \
lesson.

Rules for the lesson:
- Write it for a future run working on a DIFFERENT dataset. No run ids, no \
experiment numbers, no "in this run".
- State the condition it applies under, not just the action. "Weighted \
sampling helps when the minority class is under 5% of samples" is reusable; \
"weighted sampling helped" is not.
- If the intervention failed, still write the lesson, and set avoid=true. \
Negative results are the ones that save a future run the most time.
- Set reusable=false when the change was within evaluation noise, so it is \
recorded but never retrieved as guidance.
- Do not credit an intervention for an improvement it cannot explain. If macro \
F1 rose but the targeted class did not, say so.
"""


def critique_result(
    client: FireworksClient,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    diagnosis: Mapping[str, Any] | None = None,
    dataset_summary: Mapping[str, Any] | None = None,
    noise_threshold: float = 0.01,
    role: Role = Role.CRITIC,
) -> dict[str, Any]:
    """Frozen interface: `critique_result(before, after, action) -> dict`.

    `before` and `after` are metric dicts (macro_f1, per_class, ...). `action`
    is the executed proposal. Returns a dict matching `CRITIQUE_SCHEMA`, with a
    `lesson` sub-object ready for the MongoDB `lessons` collection.
    """
    delta = _delta(before, after)
    prompt = _render_critique_prompt(
        before, after, action, delta, diagnosis, dataset_summary, noise_threshold
    )

    result = complete_json(
        client,
        name="Critique",
        schema=CRITIQUE_SCHEMA,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        role=role,
        temperature=client.settings.critic_temperature,
        purpose="critique",
    )

    # The verdict is arithmetic, not opinion. Overwrite whatever the model said
    # so downstream stop conditions cannot be fooled by a generous critic.
    measured = _verdict(delta, noise_threshold)
    if result.get("verdict") != measured:
        log.info("critic said verdict=%s; measured delta says %s. Using measured.",
                 result.get("verdict"), measured)
        result["verdict"] = measured

    lesson = result.setdefault("lesson", {})
    lesson.setdefault("avoid", measured in ("regressed", "no_change"))
    if measured == "no_change" and abs(delta.get("macro_f1", 0.0)) < noise_threshold:
        lesson["reusable"] = False

    result["measured_delta"] = delta
    result["calibration"] = _calibration(action, delta)
    return result


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    """Numeric before/after diff for every shared scalar metric."""
    out: dict[str, float] = {}
    for key, new_value in after.items():
        old_value = before.get(key)
        if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
            if not isinstance(new_value, bool) and not isinstance(old_value, bool):
                out[key] = round(float(new_value) - float(old_value), 6)

    per_class_before = before.get("per_class") or {}
    per_class_after = after.get("per_class") or {}
    if isinstance(per_class_before, Mapping) and isinstance(per_class_after, Mapping):
        for cls, metrics in per_class_after.items():
            old = per_class_before.get(cls)
            if isinstance(metrics, Mapping) and isinstance(old, Mapping):
                for metric, value in metrics.items():
                    prev = old.get(metric)
                    if isinstance(value, (int, float)) and isinstance(prev, (int, float)):
                        out[f"per_class.{cls}.{metric}"] = round(
                            float(value) - float(prev), 6
                        )
    return out


def _verdict(delta: Mapping[str, float], noise_threshold: float) -> str:
    primary = delta.get("macro_f1")
    if primary is None:
        primary = delta.get("f1") or delta.get("accuracy")
    if primary is None:
        return "mixed"
    if primary > noise_threshold:
        return "improved"
    if primary < -noise_threshold:
        return "regressed"
    return "no_change"


def _calibration(action: Mapping[str, Any], delta: Mapping[str, float]) -> dict[str, Any]:
    """How close the scientist's predicted delta was to reality.

    Tracked across a run so we can show the agent's forecasts getting better,
    which is a second, independent piece of evidence for "the agent improved".
    """
    predicted = action.get("expected_delta_macro_f1")
    actual = delta.get("macro_f1")
    if not isinstance(predicted, (int, float)) or not isinstance(actual, (int, float)):
        return {"predicted": predicted, "actual": actual, "error": None}
    return {
        "predicted": round(float(predicted), 4),
        "actual": round(float(actual), 4),
        "error": round(abs(float(predicted) - float(actual)), 4),
        "direction_correct": (predicted >= 0) == (actual >= 0),
    }


def _render_critique_prompt(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    action: Mapping[str, Any],
    delta: Mapping[str, float],
    diagnosis: Mapping[str, Any] | None,
    dataset_summary: Mapping[str, Any] | None,
    noise_threshold: float,
) -> str:
    import json

    parts = []
    if dataset_summary:
        parts.append("DATASET\n" + json.dumps(dataset_summary, indent=2, default=str))
    if diagnosis:
        parts.append(
            "THE HYPOTHESIS UNDER TEST\n"
            f"diagnosis:  {diagnosis.get('diagnosis', '')}\n"
            f"hypothesis: {diagnosis.get('hypothesis', '')}"
        )
    parts.append("INTERVENTION APPLIED\n" + json.dumps(action, indent=2, default=str))
    parts.append("METRICS BEFORE\n" + json.dumps(before, indent=2, default=str))
    parts.append("METRICS AFTER\n" + json.dumps(after, indent=2, default=str))
    parts.append("MEASURED DELTA\n" + json.dumps(_top_deltas(delta), indent=2))
    parts.append(
        f"Changes smaller than {noise_threshold} in absolute macro F1 are within "
        "evaluation noise and must not be described as an improvement."
    )
    parts.append(
        "TASK\nJudge the hypothesis, explain the outcome, and write one reusable "
        "lesson for a future run on a different dataset."
    )
    return "\n\n".join(parts)


def _top_deltas(delta: Mapping[str, float], limit: int = 12) -> dict[str, float]:
    """Keep the prompt readable on datasets with many classes."""
    if len(delta) <= limit:
        return dict(delta)
    ranked = sorted(delta.items(), key=lambda kv: abs(kv[1]), reverse=True)
    headline = {k: v for k, v in delta.items() if "." not in k}
    for key, value in ranked:
        if len(headline) >= limit:
            break
        headline.setdefault(key, value)
    return headline


def to_lesson_document(
    critique: Mapping[str, Any],
    *,
    experiment_id: str,
    run_id: str,
    failure_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shape the critique into the `lessons` document from the team plan.

    Person 1 adds `_id`, the embedding vector, and timestamps on write; this
    function only guarantees the fields exist and are consistent.
    """
    lesson = dict(critique.get("lesson") or {})
    delta = critique.get("measured_delta") or {}
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "failure_summary": lesson.get("failure_summary", ""),
        "intervention": lesson.get("intervention", ""),
        "result": lesson.get("result", ""),
        "lesson": lesson.get("lesson", ""),
        "reusable": bool(lesson.get("reusable", True)),
        "avoid": bool(lesson.get("avoid", False)),
        "confidence": float(critique.get("confidence", 0.5)),
        "verdict": critique.get("verdict"),
        "delta_macro_f1": delta.get("macro_f1"),
        "failure_mode": (failure_signature or {}).get("failure_mode"),
        "affected_classes": (failure_signature or {}).get("affected_classes", []),
        # `embedding_text` is exactly what gets vectorised. Storing it makes the
        # retrieval reproducible and lets you re-embed after a model change
        # without re-deriving the string.
        "embedding_text": embedding_text(lesson, failure_signature),
    }


def embedding_text(
    lesson: Mapping[str, Any], failure_signature: Mapping[str, Any] | None = None
) -> str:
    """The canonical text vectorised for Atlas Vector Search.

    Query side embeds `failure_signature.summary`; document side embeds this.
    Both must describe the *problem* in the same register or similarity search
    degrades into matching on the intervention wording instead.
    """
    parts = [lesson.get("failure_summary", "")]
    if failure_signature:
        mode = failure_signature.get("failure_mode")
        classes = failure_signature.get("affected_classes") or []
        if mode:
            parts.append(f"failure mode: {mode}")
        if classes:
            parts.append(f"affected classes: {', '.join(str(c) for c in classes)}")
    parts.append(lesson.get("lesson", ""))
    return "\n".join(p for p in parts if p).strip()
