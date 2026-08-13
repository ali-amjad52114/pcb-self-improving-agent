"""Deterministic fake ML pipeline — deltas depend on selected action."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ACTION_EFFECTS: dict[str, float] = {
    "change_learning_rate": 0.06,
    "change_batch_size": 0.08,
    "change_sampler": 0.09,
    "change_class_weights": 0.06,
    "change_augmentation": 0.03,
    "change_image_size": 0.02,
    "change_confidence_threshold": 0.01,
    "change_model": 0.05,
}

# When FakeReasoning marks memory_confirmed=True, amplify the measured effect.
# This models "we already know this intervention works for this failure signature"
# without a hidden warm_run boolean.
MEMORY_CONFIRMED_BONUS: dict[str, float] = {
    "change_sampler": 0.09,  # 0.09 + 0.09 = 0.18
    "change_class_weights": 0.05,  # 0.06 + 0.05 = 0.11
    "change_learning_rate": 0.02,
    "change_batch_size": 0.0,
}


# Intentionally poor parameter choices (for memory "skip bad ideas" demos).
BAD_PARAMETER_HINTS = {
    ("change_batch_size", "batch_size", 128): -0.01,
    ("change_learning_rate", "learning_rate", 0.01): -0.02,
}


class FakeML:
    """Person 2 ML contract with action-dependent metric outcomes."""

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}
        self._train_count = 0

    def train_model(self, config: dict) -> dict:
        self._train_count += 1
        # Deterministic IDs: prefer orchestrator experiment_id for retry safety.
        experiment_id = config.get("_experiment_id")
        if experiment_id:
            model_id = f"model_{experiment_id}"
        else:
            model_id = f"model_fake_{self._train_count:03d}"
        stored = {
            "model_id": model_id,
            "config": deepcopy(config),
            "last_action": config.get("_last_action"),
            "last_parameters": deepcopy(config.get("_last_parameters") or {}),
        }
        self._models[model_id] = stored
        return {"model_id": model_id, "metadata": {"source": "fake_ml"}}

    def evaluate_model(self, model_id: str) -> dict:
        model = self._models.get(model_id)
        if model is None:
            macro_f1 = 0.55
            action = None
            params: dict[str, Any] = {}
            config: dict[str, Any] = {}
        else:
            action = model.get("last_action")
            params = model.get("last_parameters") or {}
            config = model.get("config") or {}
            macro_f1 = self._score_for(action, params, config)
            # Persist cumulative base onto stored config for the next apply_experiment.
            model["config"]["_macro_f1_base"] = macro_f1

        open_circuit_recall = 0.41
        if action == "change_sampler":
            open_circuit_recall = min(0.95, 0.41 + 0.35)
        elif action == "change_class_weights":
            open_circuit_recall = min(0.95, 0.41 + 0.22)
        elif action == "change_batch_size":
            open_circuit_recall = 0.45
        elif action == "change_learning_rate":
            open_circuit_recall = 0.48

        accuracy = min(0.99, macro_f1 + 0.04)
        return {
            "metrics": {
                "macro_f1": round(macro_f1, 4),
                "accuracy": round(accuracy, 4),
            },
            "confusion_matrix": {
                "open_circuit": {"open_circuit": 40, "short": 8, "spurious": 12},
                "short": {"open_circuit": 3, "short": 55, "spurious": 2},
                "spurious": {"open_circuit": 5, "short": 4, "spurious": 50},
            },
            "per_class_metrics": {
                "open_circuit": {
                    "recall": round(open_circuit_recall, 4),
                    "precision": 0.72,
                },
                "short": {"recall": 0.88, "precision": 0.85},
                "spurious": {"recall": 0.80, "precision": 0.78},
            },
        }

    def apply_experiment(self, current_config: dict, proposal: dict) -> dict:
        new_config = deepcopy(current_config)
        action = proposal.get("next_action")
        parameters = dict(proposal.get("parameters") or {})
        new_config["_last_action"] = action
        new_config["_last_parameters"] = parameters
        # Carry cumulative score base from prior evaluated config.
        new_config["_macro_f1_base"] = float(current_config.get("_macro_f1_base", 0.55))

        if action == "change_learning_rate":
            new_config["learning_rate"] = parameters.get(
                "learning_rate", float(new_config.get("learning_rate", 0.001)) * 0.3
            )
        elif action == "change_batch_size":
            new_config["batch_size"] = parameters.get("batch_size", 64)
        elif action == "change_sampler":
            new_config["sampler"] = parameters.get("strategy", "weighted")
        elif action == "change_class_weights":
            new_config["class_weights"] = parameters.get("weights", "balanced")
        elif action == "change_augmentation":
            new_config["augmentation"] = parameters.get("augmentation", "strong")
        elif action == "change_image_size":
            new_config["image_size"] = parameters.get("image_size", 256)
        elif action == "change_confidence_threshold":
            new_config["confidence_threshold"] = parameters.get("threshold", 0.4)
        elif action == "change_model":
            new_config["model_family"] = parameters.get(
                "model_family", "efficientnet_b0"
            )

        return new_config

    def _score_for(
        self,
        action: str | None,
        params: dict[str, Any],
        config: dict[str, Any],
    ) -> float:
        base = float(config.get("_macro_f1_base", 0.55))
        if not action:
            return base

        delta = ACTION_EFFECTS.get(action, 0.0)
        for (bad_action, key, bad_value), penalty in BAD_PARAMETER_HINTS.items():
            if action == bad_action and params.get(key) == bad_value:
                delta = penalty
                break

        if params.get("memory_confirmed") and delta > 0:
            delta += MEMORY_CONFIRMED_BONUS.get(action, 0.0)

        return round(min(0.95, max(0.3, base + delta)), 4)
