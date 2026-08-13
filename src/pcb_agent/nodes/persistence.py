"""Store experiment/lesson experience and update iteration history."""

from __future__ import annotations

from typing import Any, Callable

from pcb_agent import console
from pcb_agent.contracts import metric_value
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.routing import stop_reason_for
from pcb_agent.state import AgentState


def make_store_experience(
    deps: AgentDependencies,
) -> Callable[[AgentState], dict[str, Any]]:
    def store_experience(state: AgentState) -> dict[str, Any]:
        warnings = list(state.get("warnings") or [])
        iteration = int(state.get("iteration") or 0)
        next_iteration = iteration + 1
        run_id = str(state.get("run_id") or "run")
        experiment_id = str(
            state.get("pending_experiment_id")
            or f"{run_id}_exp_{next_iteration:03d}"
        )
        proposal = dict(state.get("proposed_experiment") or {})
        critique = dict(state.get("critique") or {})
        lesson_body = dict(critique.get("lesson") or {})
        helped = bool(state.get("last_experiment_helped"))
        target_metric = state.get("target_metric") or "macro_f1"
        before = metric_value(state.get("previous_metrics"), target_metric)
        after = metric_value(state.get("current_metrics"), target_metric)
        delta = float(state.get("last_metric_delta") or (after - before))

        experiment_payload = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "iteration": next_iteration,
            "dataset_stats": state.get("dataset_summary") or {},
            "model_config": state.get("current_config") or {},
            "metrics": state.get("current_metrics") or {},
            "previous_metrics": state.get("previous_metrics") or {},
            "failure_signature": {
                "per_class_metrics": state.get("per_class_metrics") or {},
                "memory_query": state.get("memory_query") or "",
            },
            "intervention": proposal,
            "outcome_delta": {target_metric: delta},
            "critic": critique,
            "independent_evaluation": state.get("evaluator_opinion") or {},
            "status": "complete",
        }

        try:
            deps.memory.store_experiment(experiment_payload)
            experiment_stored = True
        except Exception as exc:
            warnings.append(f"store_experiment_failed: {exc}")
            experiment_stored = False

        lesson_payload = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "failure_summary": lesson_body.get("failure_summary")
            or "Poor recall on minority open-circuit class",
            "intervention": lesson_body.get("intervention")
            or proposal.get("next_action")
            or "",
            "result": lesson_body.get("result")
            or f"{'+' if delta >= 0 else ''}{delta:.2f} macro F1",
            "confidence": float(lesson_body.get("confidence") or 0.0),
            "helped": helped,
            "next_action": proposal.get("next_action"),
        }

        lesson_ok = False
        try:
            deps.memory.store_lesson(lesson_payload)
            lesson_ok = True
        except Exception as exc:
            # Recoverable if experiment record succeeded.
            warnings.append(f"store_lesson_failed: {exc}")
            if not experiment_stored:
                warnings.append("memory_storage_failed_entirely")

        retrieved = state.get("retrieved_lessons") or []
        memory_used = [
            str(lesson.get("lesson_id") or lesson.get("experiment_id") or idx)
            for idx, lesson in enumerate(retrieved)
        ]

        history = list(state.get("experiment_history") or [])
        history.append(
            {
                "experiment_id": experiment_id,
                "iteration": next_iteration,
                "proposal": {
                    "next_action": proposal.get("next_action"),
                    "parameters": proposal.get("parameters") or {},
                },
                f"before_{target_metric}": before,
                f"after_{target_metric}": after,
                "delta": delta,
                "helped": helped,
                "memory_used": memory_used,
            }
        )

        console.print_memory_write(experiment_id, lesson_ok)

        updates: dict[str, Any] = {
            "experiment_history": history,
            "iteration": next_iteration,
            "proposal_revision_count": 0,
            "warnings": warnings,
            "pending_experiment_id": experiment_id,
            "status": "experiment_stored",
        }

        # Apply stop metadata if this iteration ends the run.
        probe = {**state, **updates}
        reason = stop_reason_for(probe)  # type: ignore[arg-type]
        if reason:
            updates["status"] = "complete"
            updates["stop_reason"] = reason

        return updates

    return store_experience
