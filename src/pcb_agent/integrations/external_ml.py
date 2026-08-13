"""Dynamic loader for Person 2 ML module."""

from __future__ import annotations

import importlib
from typing import Any

from pcb_agent.contracts import IntegrationError

REQUIRED = (
    "train_model",
    "evaluate_model",
    "apply_experiment",
)


class ExternalMLAdapter:
    def __init__(self, module: Any) -> None:
        self._module = module

    def train_model(self, config: dict) -> dict:
        return self._module.train_model(config)

    def evaluate_model(self, model_id: str) -> dict:
        return self._module.evaluate_model(model_id)

    def apply_experiment(self, current_config: dict, proposal: dict) -> dict:
        return self._module.apply_experiment(current_config, proposal)


def load_ml_adapter(module_path: str) -> ExternalMLAdapter:
    if not module_path:
        raise IntegrationError(
            "IntegrationError:\n"
            "ML_MODULE is empty. Set ML_MODULE to Person 2's module path "
            "or use AGENT_MODE=fake."
        )
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise IntegrationError(
            f"IntegrationError:\nML_MODULE={module_path} could not be imported: {exc}"
        ) from exc

    for name in REQUIRED:
        if not callable(getattr(module, name, None)):
            raise IntegrationError(
                f"IntegrationError:\n"
                f"ML_MODULE={module_path} is missing required function {name}()"
            )
    return ExternalMLAdapter(module)
