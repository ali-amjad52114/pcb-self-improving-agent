"""Memory retrieval node."""

from __future__ import annotations

from typing import Any, Callable

from pcb_agent import console
from pcb_agent.config import Settings
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.memory_query import build_memory_query
from pcb_agent.state import AgentState


def make_retrieve_memory(
    deps: AgentDependencies,
    settings: Settings,
) -> Callable[[AgentState], dict[str, Any]]:
    def retrieve_memory(state: AgentState) -> dict[str, Any]:
        warnings = list(state.get("warnings") or [])
        query = build_memory_query(state)
        lessons: list[dict[str, Any]] = []
        try:
            raw = deps.memory.retrieve_similar_lessons(
                query=query,
                k=settings.memory_top_k,
            )
            lessons = [dict(x) for x in (raw or [])]
        except Exception as exc:  # recoverable
            warnings.append(f"memory_retrieval_failed: {exc}")
            lessons = []

        iteration = int(state.get("iteration") or 0)
        console.print_iteration_header(iteration + 1)
        console.print_memory(query, lessons)
        return {
            "memory_query": query,
            "retrieved_lessons": lessons,
            "warnings": warnings,
            "judge_skipped": False,
            "evaluator_trigger": "",
        }

    return retrieve_memory
