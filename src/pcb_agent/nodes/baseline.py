"""Baseline train + evaluate nodes."""

from __future__ import annotations

from typing import Any, Callable

from pcb_agent import console
from pcb_agent.contracts import flatten_metrics, metric_value, normalize_metrics
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.state import AgentState


def make_train_baseline(deps: AgentDependencies) -> Callable[[AgentState], dict[str, Any]]:
    def train_baseline(state: AgentState) -> dict[str, Any]:
        result = deps.ml.train_model(dict(state.get("current_config") or {}))
        model_id = str(result.get("model_id") or "")
        if not model_id:
            raise RuntimeError("train_model() returned no model_id")
        return {
            "current_model_id": model_id,
            "status": "baseline_trained",
        }

    return train_baseline


def make_evaluate_baseline(
    deps: AgentDependencies,
) -> Callable[[AgentState], dict[str, Any]]:
    def evaluate_baseline(state: AgentState) -> dict[str, Any]:
        raw = deps.ml.evaluate_model(state["current_model_id"])
        normalized = normalize_metrics(raw)
        metrics = flatten_metrics(raw)
        config = dict(state.get("current_config") or {})
        # Seed cumulative fake score base after baseline eval.
        config["_macro_f1_base"] = metric_value(metrics, state.get("target_metric") or "macro_f1")
        updates: dict[str, Any] = {
            "baseline_metrics": metrics,
            "current_metrics": metrics,
            "best_metrics": metrics,
            "previous_metrics": metrics,
            "best_config": config,
            "current_config": config,
            "confusion_matrix": normalized["confusion_matrix"],
            "per_class_metrics": normalized["per_class_metrics"],
            "training_history": normalized["training_history"],
            "misclassified_examples": normalized["misclassified_examples"],
            "status": "baseline_evaluated",
            "iteration": int(state.get("iteration") or 0),
            "experiment_history": list(state.get("experiment_history") or []),
            "warnings": list(state.get("warnings") or []),
            "consecutive_failures": int(state.get("consecutive_failures") or 0),
            "proposal_revision_count": 0,
            "openrouter_calls": int(state.get("openrouter_calls") or 0),
        }
        merged = {**state, **updates}
        console.print_baseline(merged)  # type: ignore[arg-type]
        return updates

    return evaluate_baseline
