"""Evidence-driven scientist API and compatibility wrappers."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from fireworks.client import FireworksClient
from fireworks.prompts import DIAGNOSIS_SYSTEM_PROMPT, SCIENTIST_SYSTEM_PROMPT
from fireworks.schemas import Diagnosis, ExperimentProposal


T = TypeVar("T", bound=BaseModel)


class StructuredGenerator(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        payload: dict,
        response_model: type[T],
        schema_name: str,
    ) -> T: ...


def diagnose_and_propose(
    current_config: dict,
    metrics: dict,
    per_class_metrics: dict | None = None,
    confusion_matrix: list | dict | None = None,
    dataset_summary: dict | None = None,
    recent_experiments: list | None = None,
    retrieved_memories: list | None = None,
    *,
    client: StructuredGenerator | None = None,
) -> ExperimentProposal:
    """Analyze evidence and return exactly one validated experiment proposal."""
    generator = client or FireworksClient()
    payload = {
        "current_config": current_config or {},
        "metrics": metrics or {},
        "per_class_metrics": per_class_metrics or {},
        "confusion_matrix": confusion_matrix or [],
        "dataset_summary": dataset_summary or {},
        "recent_experiments": recent_experiments or [],
        "retrieved_memories": _label_memories(retrieved_memories or []),
    }
    proposal = generator.generate(
        system_prompt=SCIENTIST_SYSTEM_PROMPT,
        payload=payload,
        response_model=ExperimentProposal,
        schema_name="pcb_experiment_proposal",
    )
    presented_ids = {
        str(item["memory_id"])
        for item in payload["retrieved_memories"]
        if item.get("memory_id")
    }
    return proposal.model_copy(
        update={
            "memory_used": [
                memory_id
                for memory_id in proposal.memory_used
                if memory_id in presented_ids
            ]
        }
    )


def diagnose(state: dict, memories: list) -> dict[str, Any]:
    """Compatibility function for the existing separate diagnosis contract."""
    generator = FireworksClient()
    payload = {
        "current_config": state.get("current_config") or state.get("config") or {},
        "metrics": state.get("current_metrics") or state.get("metrics") or {},
        "per_class_metrics": state.get("per_class_metrics") or {},
        "confusion_matrix": state.get("confusion_matrix") or [],
        "dataset_summary": state.get("dataset_summary") or {},
        "recent_experiments": state.get("experiment_history") or [],
        "retrieved_memories": _label_memories(memories or []),
    }
    result = generator.generate(
        system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
        payload=payload,
        response_model=Diagnosis,
        schema_name="pcb_failure_diagnosis",
    )
    return result.model_dump()


def propose_experiment(state: dict, diagnosis: dict) -> dict[str, Any]:
    """Compatibility function returning the existing orchestrator proposal shape."""
    recent_experiments = list(state.get("experiment_history") or [])
    evaluator_opinion = state.get("evaluator_opinion") or {}
    if evaluator_opinion.get("recommendation") == "reconsider":
        recent_experiments.append(
            {
                "type": "independent_evaluator_feedback",
                "proposal_to_revise": state.get("proposed_experiment") or {},
                "feedback": evaluator_opinion,
                "instruction": "Choose a materially revised supported experiment.",
            }
        )
    proposal = diagnose_and_propose(
        current_config=state.get("current_config") or state.get("config") or {},
        metrics=state.get("current_metrics") or state.get("metrics") or {},
        per_class_metrics=state.get("per_class_metrics") or {},
        confusion_matrix=state.get("confusion_matrix") or [],
        dataset_summary={
            **(state.get("dataset_summary") or {}),
            "prior_diagnosis": diagnosis,
        },
        recent_experiments=recent_experiments,
        retrieved_memories=state.get("retrieved_lessons") or [],
    )
    return proposal.as_orchestrator_proposal()


def _label_memories(memories: list) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for index, memory in enumerate(memories):
        if isinstance(memory, dict):
            item = dict(memory)
        else:
            item = {"lesson": str(memory)}
        item.setdefault(
            "memory_id",
            str(
                item.get("lesson_id")
                or item.get("_id")
                or item.get("experiment_id")
                or f"memory_{index}"
            ),
        )
        labeled.append(item)
    return labeled
