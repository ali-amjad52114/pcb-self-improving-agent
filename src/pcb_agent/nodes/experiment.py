"""Run experiment + evaluate result nodes."""

from __future__ import annotations

from typing import Any, Callable

from pcb_agent import console
from pcb_agent.contracts import METRIC_EPSILON, flatten_metrics, metric_value, normalize_metrics
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.state import AgentState


def make_experiment_id(run_id: str, iteration: int) -> str:
    """Deterministic experiment ID for retry/idempotency (iteration is 0-based completed count)."""
    # Experiment N uses iteration+1 because iteration counts completed experiments.
    return f"{run_id}_exp_{iteration + 1:03d}"


def make_run_experiment(
    deps: AgentDependencies,
) -> Callable[[AgentState], dict[str, Any]]:
    def run_experiment(state: AgentState) -> dict[str, Any]:
        iteration = int(state.get("iteration") or 0)
        run_id = str(state.get("run_id") or "run")
        # Deterministic ID — do not regenerate on retry of the same checkpointed step.
        experiment_id = make_experiment_id(run_id, iteration)

        proposal = dict(state.get("proposed_experiment") or {})
        current_config = dict(state.get("current_config") or {})
        new_config = deps.ml.apply_experiment(current_config, proposal)
        # Attach idempotency keys for Person 2 if they later honor them.
        new_config["_experiment_id"] = experiment_id
        new_config["_run_id"] = run_id
        new_config["_iteration"] = iteration + 1

        train_result = deps.ml.train_model(new_config)
        model_id = str(train_result.get("model_id") or "")
        if not model_id:
            raise RuntimeError("train_model() returned no model_id during experiment")

        return {
            "previous_metrics": dict(state.get("current_metrics") or {}),
            "previous_confusion_matrix": state.get("confusion_matrix") or {},
            "previous_per_class_metrics": dict(state.get("per_class_metrics") or {}),
            "current_config": new_config,
            "current_model_id": model_id,
            "pending_experiment_id": experiment_id,
            "status": "experiment_trained",
        }

    return run_experiment


def make_evaluate_result(
    deps: AgentDependencies,
) -> Callable[[AgentState], dict[str, Any]]:
    def evaluate_result(state: AgentState) -> dict[str, Any]:
        raw = deps.ml.evaluate_model(state["current_model_id"])
        normalized = normalize_metrics(raw)
        metrics = flatten_metrics(raw)

        target_metric = state.get("target_metric") or "macro_f1"
        previous = dict(state.get("previous_metrics") or {})
        before = metric_value(previous, target_metric)
        after = metric_value(metrics, target_metric)
        delta = after - before
        helped = delta > METRIC_EPSILON

        best_metrics = dict(state.get("best_metrics") or {})
        best_config = dict(state.get("best_config") or {})
        best_value = metric_value(best_metrics, target_metric)
        if after > best_value + METRIC_EPSILON:
            best_metrics = metrics
            best_config = dict(state.get("current_config") or {})

        consecutive = int(state.get("consecutive_failures") or 0)
        consecutive = 0 if helped else consecutive + 1

        # Keep fake cumulative base in sync for next apply_experiment.
        current_config = dict(state.get("current_config") or {})
        current_config["_macro_f1_base"] = after

        console.print_result(target_metric, before, after, delta)

        return {
            "current_metrics": metrics,
            "current_config": current_config,
            "confusion_matrix": normalized["confusion_matrix"],
            "per_class_metrics": normalized["per_class_metrics"],
            "training_history": normalized["training_history"],
            "misclassified_examples": normalized["misclassified_examples"],
            "last_metric_delta": delta,
            "last_experiment_helped": helped,
            "best_metrics": best_metrics,
            "best_config": best_config,
            "consecutive_failures": consecutive,
            "status": "experiment_evaluated",
        }

    return evaluate_result
