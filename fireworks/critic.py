"""Experiment critic API and compatibility wrapper."""

from __future__ import annotations

from typing import Any

from fireworks.client import FireworksClient
from fireworks.prompts import CRITIC_SYSTEM_PROMPT
from fireworks.schemas import ExperimentCritique
from fireworks.scientist import StructuredGenerator


def critique_experiment(
    before: dict,
    experiment: dict,
    after: dict,
    *,
    client: StructuredGenerator | None = None,
) -> ExperimentCritique:
    """Evaluate measured results and return a validated reusable lesson."""
    generator = client or FireworksClient()
    critique = generator.generate(
        system_prompt=CRITIC_SYSTEM_PROMPT,
        payload={
            "before": before or {},
            "experiment": experiment or {},
            "after": after or {},
        },
        response_model=ExperimentCritique,
        schema_name="pcb_experiment_critique",
    )
    return _apply_measured_metrics(critique, before, after)


def critique_result(before: dict, after: dict, action: dict) -> dict[str, Any]:
    """Compatibility function returning the existing orchestrator critique shape."""
    critique = critique_experiment(before, action, after)
    helped = critique.verdict in {"successful", "partially_successful"}
    action_name = str(action.get("next_action") or action.get("action") or "unknown")
    delta_text = ", ".join(
        f"{name} {value:+.4f}" for name, value in critique.metric_delta.items()
    ) or "no comparable metric delta"
    return {
        "helped": helped,
        "reason": critique.analysis,
        "lesson": {
            "failure_summary": critique.lesson,
            "intervention": action_name,
            "result": delta_text,
            "confidence": critique.confidence,
            "next_action": action_name,
            "helped": helped,
        },
    }


def _apply_measured_metrics(
    critique: ExperimentCritique,
    before: dict,
    after: dict,
) -> ExperimentCritique:
    """Keep metric arithmetic and obvious verdict contradictions deterministic."""
    before_metrics = before.get("metrics", before) if isinstance(before, dict) else {}
    after_metrics = after.get("metrics", after) if isinstance(after, dict) else {}
    deltas = {
        name: round(float(after_metrics[name]) - float(value), 10)
        for name, value in before_metrics.items()
        if name in after_metrics
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isinstance(after_metrics[name], (int, float))
        and not isinstance(after_metrics[name], bool)
    }
    updates: dict[str, Any] = {"metric_delta": deltas}
    if not deltas:
        updates.update(verdict="inconclusive", hypothesis_supported=False)
    elif (
        deltas.get("macro_f1", deltas.get("accuracy", 0.0)) <= 0
        and critique.verdict == "successful"
    ):
        updates.update(verdict="failed", hypothesis_supported=False)
    return critique.model_copy(update=updates)
