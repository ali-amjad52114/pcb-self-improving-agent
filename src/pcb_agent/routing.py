"""Pure routing helpers for LangGraph conditional edges."""

from __future__ import annotations

from pcb_agent.contracts import metric_value
from pcb_agent.state import AgentState

JUDGE_CONFIDENCE_THRESHOLD = 0.65


def proposal_judge_reason(state: AgentState) -> str | None:
    """Return a trigger reason when OpenRouter/judge should review the proposal."""
    diagnosis = state.get("diagnosis") or {}
    proposal = state.get("proposed_experiment") or {}

    try:
        diagnosis_confidence = float(diagnosis.get("confidence", 1.0))
    except (TypeError, ValueError):
        diagnosis_confidence = 1.0
    if diagnosis_confidence < JUDGE_CONFIDENCE_THRESHOLD:
        return "low_diagnosis_confidence"

    try:
        proposal_confidence = float(proposal.get("confidence", 1.0))
    except (TypeError, ValueError):
        proposal_confidence = 1.0
    if proposal_confidence < JUDGE_CONFIDENCE_THRESHOLD:
        return "low_proposal_confidence"

    if int(state.get("consecutive_failures") or 0) >= 2:
        return "two_consecutive_failures"

    if proposal.get("next_action") == "change_model":
        return "model_family_switch"

    return None


def should_reconsider(state: AgentState) -> bool:
    """True when judge says reconsider and we have not revised yet this iteration."""
    opinion = state.get("evaluator_opinion") or {}
    revision_count = int(state.get("proposal_revision_count") or 0)
    return (
        opinion.get("recommendation") == "reconsider"
        and revision_count < 1
    )


def will_stop_after_iteration(state: AgentState) -> bool:
    """Predict whether store_experience will end the run after this experiment."""
    # After store, iteration becomes current+1; stop checks use the incremented value.
    next_iteration = int(state.get("iteration") or 0) + 1
    budget = int(state.get("experiment_budget") or 0)
    target_metric = state.get("target_metric") or "macro_f1"
    target_value = float(state.get("target_value") or 0.0)
    current = metric_value(state.get("current_metrics"), target_metric)
    if current >= target_value:
        return True
    if next_iteration >= budget:
        return True
    return False


def stop_reason_for(state: AgentState) -> str | None:
    """Return stop_reason if the run should end given *post-store* state."""
    target_metric = state.get("target_metric") or "macro_f1"
    target_value = float(state.get("target_value") or 0.0)
    current = metric_value(state.get("current_metrics"), target_metric)
    iteration = int(state.get("iteration") or 0)
    budget = int(state.get("experiment_budget") or 0)

    if current >= target_value:
        return "target_reached"
    if iteration >= budget:
        return "experiment_budget_exhausted"
    return None


def should_continue(state: AgentState) -> str:
    """LangGraph routing: 'end' or 'retrieve_memory'."""
    reason = stop_reason_for(state)
    if reason:
        return "end"
    return "retrieve_memory"


def route_after_propose(state: AgentState) -> str:
    if proposal_judge_reason(state):
        return "judge_proposal"
    return "run_experiment"


def route_after_judge_proposal(state: AgentState) -> str:
    if should_reconsider(state):
        return "propose_experiment"
    return "run_experiment"


def route_after_critique(state: AgentState, *, openrouter_enabled: bool) -> str:
    if openrouter_enabled and will_stop_after_iteration(state):
        return "judge_lesson"
    return "store_experience"
