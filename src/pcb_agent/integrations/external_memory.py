"""Dynamic loader for Person 1 memory module."""

from __future__ import annotations

import importlib
from typing import Any

from pcb_agent.contracts import IntegrationError

REQUIRED = (
    "retrieve_similar_lessons",
    "store_experiment",
    "store_lesson",
)


class ExternalMemoryAdapter:
    def __init__(self, module: Any) -> None:
        self._module = module

    def retrieve_similar_lessons(self, query: str, k: int) -> list:
        return self._module.retrieve_similar_lessons(query, k)

    def store_experiment(self, data: dict) -> str:
        return self._module.store_experiment(data)

    def store_lesson(self, data: dict) -> str:
        return self._module.store_lesson(data)

    def get_run_history(self, run_id: str) -> list:
        if hasattr(self._module, "get_run_history"):
            return self._module.get_run_history(run_id)
        return []


def load_memory_adapter(module_path: str) -> ExternalMemoryAdapter:
    if not module_path:
        raise IntegrationError(
            "IntegrationError:\n"
            "MEMORY_MODULE is empty. Set MEMORY_MODULE to Person 1's module path "
            "or use AGENT_MODE=fake."
        )
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise IntegrationError(
            f"IntegrationError:\nMEMORY_MODULE={module_path} could not be imported: {exc}"
        ) from exc

    for name in REQUIRED:
        if not callable(getattr(module, name, None)):
            raise IntegrationError(
                f"IntegrationError:\n"
                f"MEMORY_MODULE={module_path} is missing required function {name}()"
            )
    return ExternalMemoryAdapter(module)
