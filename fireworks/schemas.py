"""Validated public contracts for the Fireworks reasoning layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ExperimentAction = Literal[
    "change_model",
    "change_learning_rate",
    "change_batch_size",
    "change_image_size",
    "change_augmentation",
    "change_sampler",
    "change_class_weights",
    "change_confidence_threshold",
    "increase_epochs",
]

CritiqueVerdict = Literal[
    "successful",
    "partially_successful",
    "failed",
    "inconclusive",
]

PARAMETER_KEYS: dict[str, set[str]] = {
    "change_model": {"model", "model_family"},
    "change_learning_rate": {"learning_rate"},
    "change_batch_size": {"batch_size"},
    "change_image_size": {"image_size"},
    "change_augmentation": {"augmentation"},
    "change_sampler": {"sampler", "strategy"},
    "change_class_weights": {"class_weights", "weights"},
    "change_confidence_threshold": {"confidence_threshold", "threshold"},
    "increase_epochs": {"epochs", "additional_epochs"},
}


class ExperimentProposal(BaseModel):
    """One evidence-based experiment selected from the strict allowlist."""

    model_config = ConfigDict(extra="forbid")

    diagnosis: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    action: ExperimentAction
    parameters: dict[str, Any]
    reasoning_summary: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    memory_used: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_parameters(self) -> "ExperimentProposal":
        accepted = PARAMETER_KEYS[self.action]
        if not accepted.intersection(self.parameters):
            expected = " or ".join(sorted(accepted))
            raise ValueError(f"{self.action} requires parameter {expected}")
        unexpected = set(self.parameters).difference(accepted)
        if unexpected:
            raise ValueError(
                f"{self.action} received unsupported parameters: {sorted(unexpected)}"
            )
        return self

    def as_orchestrator_proposal(self) -> dict[str, Any]:
        """Return the legacy Person 3 contract without leaking API details."""
        return {
            "diagnosis": self.diagnosis,
            "hypothesis": self.hypothesis,
            "next_action": self.action,
            "parameters": self.parameters,
            "expected_effect": self.reasoning_summary,
            "confidence": self.confidence,
            "memory_used": self.memory_used,
        }


class ExperimentCritique(BaseModel):
    """Measured evaluation and reusable lesson from a completed experiment."""

    model_config = ConfigDict(extra="forbid")

    verdict: CritiqueVerdict
    metric_delta: dict[str, float]
    hypothesis_supported: bool
    analysis: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class Diagnosis(BaseModel):
    """Compatibility schema for orchestrators that diagnose in a separate step."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1)
