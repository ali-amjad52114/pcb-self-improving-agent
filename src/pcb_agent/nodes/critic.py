"""Critique node."""

from __future__ import annotations

from typing import Any, Callable

from pcb_agent.contracts import normalize_critique
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.state import AgentState


def make_critique_result(
    deps: AgentDependencies,
) -> Callable[[AgentState], dict[str, Any]]:
    def critique_result(state: AgentState) -> dict[str, Any]:
        raw = deps.reasoning.critique_result(
            before={
                "metrics": dict(state.get("previous_metrics") or {}),
                "per_class_metrics": dict(
                    state.get("previous_per_class_metrics") or {}
                ),
                "confusion_matrix": state.get("previous_confusion_matrix") or {},
            },
            after={
                "metrics": dict(state.get("current_metrics") or {}),
                "per_class_metrics": dict(state.get("per_class_metrics") or {}),
                "confusion_matrix": state.get("confusion_matrix") or {},
            },
            action=dict(state.get("proposed_experiment") or {}),
        )
        critique = normalize_critique(
            raw,
            measured_helped=bool(state.get("last_experiment_helped")),
        )
        return {"critique": critique, "status": "critiqued"}

    return critique_result
