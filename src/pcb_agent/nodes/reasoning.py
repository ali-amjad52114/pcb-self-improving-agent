"""Scientist diagnose / propose / judge nodes."""

from __future__ import annotations

from typing import Any, Callable

from pcb_agent import console
from pcb_agent.contracts import (
    JUDGE_FALLBACK,
    normalize_diagnosis,
    normalize_judge,
    normalize_proposal,
)
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.routing import proposal_judge_reason
from pcb_agent.state import AgentState


def make_diagnose(deps: AgentDependencies) -> Callable[[AgentState], dict[str, Any]]:
    def diagnose(state: AgentState) -> dict[str, Any]:
        raw = deps.reasoning.diagnose(
            state=dict(state),
            memories=list(state.get("retrieved_lessons") or []),
        )
        diagnosis = normalize_diagnosis(raw)
        console.print_diagnosis(diagnosis)
        return {"diagnosis": diagnosis, "status": "diagnosed"}

    return diagnose


def make_propose_experiment(
    deps: AgentDependencies,
) -> Callable[[AgentState], dict[str, Any]]:
    def propose_experiment(state: AgentState) -> dict[str, Any]:
        # If routed back after judge reconsider, count this as the one allowed revision.
        revision = int(state.get("proposal_revision_count") or 0)
        opinion = state.get("evaluator_opinion") or {}
        updates: dict[str, Any] = {}
        if opinion.get("recommendation") == "reconsider" and revision < 1:
            revision = revision + 1
            updates["proposal_revision_count"] = revision

        state_for_scientist = dict(state)
        state_for_scientist["proposal_revision_count"] = revision

        raw = deps.reasoning.propose_experiment(
            state=state_for_scientist,
            diagnosis=dict(state.get("diagnosis") or {}),
        )
        proposal = normalize_proposal(raw)
        run_id = state.get("run_id")
        prior_lessons = [
            m
            for m in (state.get("retrieved_lessons") or [])
            if m.get("run_id") not in (None, "", run_id)
        ]
        memory_changed = bool(proposal.get("memory_used"))
        console.print_experiment(proposal, memory_changed=memory_changed)

        reason = proposal_judge_reason({**state, "proposed_experiment": proposal})
        updates.update(
            {
                "proposed_experiment": proposal,
                "status": "proposed",
            }
        )
        if reason is None:
            console.print_judge_skipped()
            updates["judge_skipped"] = True
            updates["evaluator_trigger"] = ""
        else:
            updates["judge_skipped"] = False
            updates["evaluator_trigger"] = reason
        return updates

    return propose_experiment


def make_judge_proposal(
    deps: AgentDependencies,
    *,
    openrouter_enabled: bool,
) -> Callable[[AgentState], dict[str, Any]]:
    def judge_proposal(state: AgentState) -> dict[str, Any]:
        reason = state.get("evaluator_trigger") or proposal_judge_reason(state) or "unknown"
        warnings = list(state.get("warnings") or [])
        calls = int(state.get("openrouter_calls") or 0)

        if not openrouter_enabled:
            opinion = dict(JUDGE_FALLBACK)
            opinion["risk_flags"] = ["openrouter_disabled"]
            opinion["reasoning_summary"] = (
                "OpenRouter disabled; proceeding with primary scientist proposal."
            )
            console.print_judge_result(reason, opinion)
            return {
                "evaluator_opinion": opinion,
                "evaluator_trigger": reason,
                "warnings": warnings,
                "openrouter_calls": calls,
            }

        history = list(state.get("experiment_history") or [])[-3:]
        context = {
            "stage": "proposal_review",
            "run_id": state.get("run_id"),
            "iteration": int(state.get("iteration") or 0) + 1,
            "dataset_summary": state.get("dataset_summary") or {},
            "current_metrics": state.get("current_metrics") or {},
            "recent_history": history,
            "retrieved_lessons": state.get("retrieved_lessons") or [],
            "diagnosis": state.get("diagnosis") or {},
            "proposal": state.get("proposed_experiment") or {},
            "trigger": reason,
        }
        try:
            opinion = normalize_judge(deps.judge.second_opinion(context))
            calls += 1
        except Exception as exc:
            warnings.append(f"openrouter_proposal_failed: {exc}")
            opinion = dict(JUDGE_FALLBACK)

        console.print_judge_result(reason, opinion)

        # Do not bump proposal_revision_count here — routing reads post-update
        # state; propose_experiment increments when it actually revises.
        return {
            "evaluator_opinion": opinion,
            "evaluator_trigger": reason,
            "warnings": warnings,
            "openrouter_calls": calls,
        }

    return judge_proposal


def make_judge_lesson(
    deps: AgentDependencies,
    *,
    openrouter_enabled: bool,
) -> Callable[[AgentState], dict[str, Any]]:
    def judge_lesson(state: AgentState) -> dict[str, Any]:
        warnings = list(state.get("warnings") or [])
        calls = int(state.get("openrouter_calls") or 0)
        if not openrouter_enabled:
            return {
                "evaluator_opinion": dict(JUDGE_FALLBACK),
                "evaluator_trigger": "final_lesson_validation",
                "warnings": warnings,
            }

        context = {
            "stage": "final_lesson_validation",
            "run_id": state.get("run_id"),
            "before_metrics": state.get("previous_metrics") or {},
            "after_metrics": state.get("current_metrics") or {},
            "action": state.get("proposed_experiment") or {},
            "critique": state.get("critique") or {},
            "retrieved_lessons": state.get("retrieved_lessons") or [],
        }
        try:
            opinion = normalize_judge(deps.judge.second_opinion(context))
            calls += 1
            console.print_judge_result("final_lesson_validation", opinion)
        except Exception as exc:
            warnings.append(f"openrouter_lesson_failed: {exc}")
            opinion = dict(JUDGE_FALLBACK)

        return {
            "evaluator_opinion": opinion,
            "evaluator_trigger": "final_lesson_validation",
            "warnings": warnings,
            "openrouter_calls": calls,
        }

    return judge_lesson
