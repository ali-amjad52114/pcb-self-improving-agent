"""Scientist diagnose / propose / judge nodes."""

from __future__ import annotations

from typing import Any, Callable

from pcb_agent import console
from pcb_agent.config import Settings
from pcb_agent.contracts import (
    JUDGE_FALLBACK,
    normalize_diagnosis,
    normalize_judge,
    normalize_proposal,
)
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.routing import proposal_judge_reason
from pcb_agent.state import AgentState


def _append_ledger(
    state: AgentState,
    *,
    deps: AgentDependencies,
    opinion: dict[str, Any],
    stage: str,
    trigger: str,
) -> list[dict[str, Any]]:
    ledger = list(state.get("openrouter_ledger") or [])
    judge_ledger = getattr(deps.judge, "ledger", None)
    if isinstance(judge_ledger, list) and judge_ledger:
        # Append only new entries from this call batch.
        calls_made = int(opinion.get("_calls_made") or 1)
        new_entries = judge_ledger[-calls_made:] if calls_made else []
        for entry in new_entries:
            row = dict(entry)
            row.setdefault("stage", stage)
            row.setdefault("trigger", trigger)
            ledger.append(row)
    else:
        panel = opinion.get("panel") or []
        if panel:
            for row in panel:
                ledger.append(
                    {
                        "model": row.get("model"),
                        "stage": stage,
                        "trigger": trigger,
                        "latency_ms": 0,
                        "tokens": None,
                        "response_format_mode": row.get("response_format_mode") or "",
                        "fallback": False,
                    }
                )
        else:
            ledger.append(
                {
                    "model": "unknown",
                    "stage": stage,
                    "trigger": trigger,
                    "latency_ms": 0,
                    "tokens": None,
                    "response_format_mode": "",
                    "fallback": "openrouter_unavailable" in (opinion.get("risk_flags") or []),
                }
            )
    return ledger


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
        memory_changed = bool(prior_lessons)
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
            updates["judge_skip_reason"] = "scientist_confidence_sufficient"
            updates["evaluator_trigger"] = ""
        else:
            updates["judge_skipped"] = False
            updates["judge_skip_reason"] = ""
            updates["evaluator_trigger"] = reason
        return updates

    return propose_experiment


def make_judge_proposal(
    deps: AgentDependencies,
    settings: Settings,
    *,
    openrouter_enabled: bool,
) -> Callable[[AgentState], dict[str, Any]]:
    def judge_proposal(state: AgentState) -> dict[str, Any]:
        reason = state.get("evaluator_trigger") or proposal_judge_reason(state) or "unknown"
        warnings = list(state.get("warnings") or [])
        calls = int(state.get("openrouter_calls") or 0)
        budget = int(
            state.get("openrouter_call_budget")
            or settings.openrouter_call_budget
            or 5
        )

        if not openrouter_enabled:
            opinion = normalize_judge(
                {
                    **JUDGE_FALLBACK,
                    "risk_flags": ["openrouter_disabled"],
                    "reasoning_summary": (
                        "OpenRouter disabled; proceeding with primary scientist proposal."
                    ),
                }
            )
            console.print_judge_result(reason, opinion, calls=calls, budget=budget)
            return {
                "evaluator_opinion": opinion,
                "evaluator_trigger": reason,
                "warnings": warnings,
                "openrouter_calls": calls,
            }

        if calls >= budget:
            opinion = normalize_judge(
                {
                    **JUDGE_FALLBACK,
                    "risk_flags": ["openrouter_budget_exhausted"],
                    "reasoning_summary": (
                        f"Independent evaluator call budget exhausted ({calls}/{budget}); "
                        "proceeding with primary scientist proposal."
                    ),
                }
            )
            console.print_judge_skipped(reason="call budget exhausted")
            return {
                "evaluator_opinion": opinion,
                "evaluator_trigger": reason,
                "warnings": warnings,
                "openrouter_calls": calls,
                "judge_skipped": True,
                "judge_skip_reason": "call_budget_exhausted",
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
            "_openrouter_calls": calls,
            "_openrouter_budget": budget,
        }
        raw: dict[str, Any] = {}
        calls_made = 0
        budget_hit = False
        try:
            raw = deps.judge.second_opinion(context)
            opinion = normalize_judge(raw)
            calls_made = int(
                raw.get("_calls_made")
                or max(1, len(opinion.get("panel") or []) or 1)
            )
            if raw.get("_budget_exhausted"):
                calls_made = 0
                budget_hit = True
                console.print_judge_skipped(reason="call budget exhausted")
            else:
                calls += calls_made
                console.print_judge_result(reason, opinion, calls=calls, budget=budget)
        except Exception as exc:
            warnings.append(f"openrouter_proposal_failed: {exc}")
            opinion = normalize_judge(dict(JUDGE_FALLBACK))
            console.print_judge_result(reason, opinion, calls=calls, budget=budget)

        ledger = _append_ledger(
            state,
            deps=deps,
            opinion={**opinion, "_calls_made": calls_made},
            stage="proposal_review",
            trigger=reason,
        )

        return {
            "evaluator_opinion": opinion,
            "evaluator_trigger": reason,
            "warnings": warnings,
            "openrouter_calls": calls,
            "openrouter_ledger": ledger,
            "judge_skipped": budget_hit,
            "judge_skip_reason": "call_budget_exhausted" if budget_hit else "",
        }

    return judge_proposal


def make_judge_lesson(
    deps: AgentDependencies,
    settings: Settings,
    *,
    openrouter_enabled: bool,
) -> Callable[[AgentState], dict[str, Any]]:
    def judge_lesson(state: AgentState) -> dict[str, Any]:
        warnings = list(state.get("warnings") or [])
        calls = int(state.get("openrouter_calls") or 0)
        budget = int(
            state.get("openrouter_call_budget")
            or settings.openrouter_call_budget
            or 5
        )
        if not openrouter_enabled:
            return {
                "evaluator_opinion": normalize_judge(
                    {**JUDGE_FALLBACK, "risk_flags": ["openrouter_disabled"]}
                ),
                "evaluator_trigger": "final_lesson_validation",
                "warnings": warnings,
            }

        if calls >= budget:
            opinion = normalize_judge(
                {
                    **JUDGE_FALLBACK,
                    "risk_flags": ["openrouter_budget_exhausted"],
                }
            )
            console.print_judge_skipped(reason="call budget exhausted")
            return {
                "evaluator_opinion": opinion,
                "evaluator_trigger": "final_lesson_validation",
                "warnings": warnings,
                "openrouter_calls": calls,
                "judge_skipped": True,
                "judge_skip_reason": "call_budget_exhausted",
            }

        context = {
            "stage": "final_lesson_validation",
            "run_id": state.get("run_id"),
            "before_metrics": state.get("previous_metrics") or {},
            "after_metrics": state.get("current_metrics") or {},
            "action": state.get("proposed_experiment") or {},
            "critique": state.get("critique") or {},
            "retrieved_lessons": state.get("retrieved_lessons") or [],
            "trigger": "final_lesson_validation",
            "_openrouter_calls": calls,
            "_openrouter_budget": budget,
        }
        try:
            raw = deps.judge.second_opinion(context)
            opinion = normalize_judge(raw)
            calls_made = int(raw.get("_calls_made") or max(1, len(opinion.get("panel") or []) or 1))
            if raw.get("_budget_exhausted"):
                calls_made = 0
                console.print_judge_skipped(reason="call budget exhausted")
            else:
                calls += calls_made
                console.print_judge_result(
                    "final_lesson_validation", opinion, calls=calls, budget=budget
                )
        except Exception as exc:
            warnings.append(f"openrouter_lesson_failed: {exc}")
            opinion = normalize_judge(dict(JUDGE_FALLBACK))
            calls_made = 0
            raw = {}

        ledger = _append_ledger(
            state,
            deps=deps,
            opinion={**opinion, "_calls_made": calls_made},
            stage="final_lesson_validation",
            trigger="final_lesson_validation",
        )

        return {
            "evaluator_opinion": opinion,
            "evaluator_trigger": "final_lesson_validation",
            "warnings": warnings,
            "openrouter_calls": calls,
            "openrouter_ledger": ledger,
        }

    return judge_lesson
