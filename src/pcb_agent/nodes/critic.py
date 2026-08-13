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
            before=dict(state.get("previous_metrics") or {}),
            after=dict(state.get("current_metrics") or {}),
            action=dict(state.get("proposed_experiment") or {}),
        )
        critique = normalize_critique(
            raw,
            measured_helped=bool(state.get("last_experiment_helped")),
        )
        return {"critique": critique, "status": "critiqued"}

    return critique_result
