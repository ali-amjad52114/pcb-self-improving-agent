"""Dynamic loader for Person 2 reasoning / Fireworks scientist module."""

from __future__ import annotations

import importlib
from typing import Any

from pcb_agent.contracts import IntegrationError

REQUIRED = (
    "diagnose",
    "propose_experiment",
    "critique_result",
)


class ExternalReasoningAdapter:
    def __init__(self, module: Any) -> None:
        self._module = module

    def diagnose(self, state: dict, memories: list) -> dict:
        return self._module.diagnose(state, memories)

    def propose_experiment(self, state: dict, diagnosis: dict) -> dict:
        return self._module.propose_experiment(state, diagnosis)

    def critique_result(self, before: dict, after: dict, action: dict) -> dict:
        return self._module.critique_result(before, after, action)


def load_reasoning_adapter(module_path: str) -> ExternalReasoningAdapter:
    if not module_path:
        raise IntegrationError(
            "IntegrationError:\n"
            "REASONING_MODULE is empty. Set REASONING_MODULE to Person 2's module path "
            "or use AGENT_MODE=fake."
        )
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise IntegrationError(
            f"IntegrationError:\nREASONING_MODULE={module_path} could not be imported: {exc}"
        ) from exc

    for name in REQUIRED:
        if not callable(getattr(module, name, None)):
            raise IntegrationError(
                f"IntegrationError:\n"
                f"REASONING_MODULE={module_path} is missing required function {name}()"
            )
    return ExternalReasoningAdapter(module)
