"""Frozen team interfaces, validation helpers, and shared exceptions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


ALLOWED_ACTIONS = {
    "change_model",
    "change_learning_rate",
    "change_batch_size",
    "change_image_size",
    "change_augmentation",
    "change_sampler",
    "change_class_weights",
    "change_confidence_threshold",
}

JUDGE_FALLBACK: dict[str, Any] = {
    "verdict": "uncertain",
    "confidence": 0.0,
    "alternative_explanation": "",
    "risk_flags": ["openrouter_unavailable"],
    "recommendation": "proceed",
    "reasoning_summary": (
        "Independent evaluator unavailable; proceeding with primary scientist proposal."
    ),
    "evidence_citations": [],
    "suggested_action_family": "",
    "suggested_action_allowed": False,
    "lesson_quality": "adequate",
    "panel": [],
    "agreement_rate": 0.0,
    "dissent_summary": "",
}

METRIC_EPSILON = 1e-6
LESSON_QUALITY_VALUES = {"weak", "adequate", "strong"}


class IntegrationError(Exception):
    """Teammate module missing a required contract function."""


class ProposalValidationError(Exception):
    """Scientist proposed an action outside the allow-list."""


class CheckpointError(Exception):
    """MongoDB checkpointing requested but unavailable."""


class JudgeOpinion(BaseModel):
    verdict: str = Field(pattern="^(agree|disagree|uncertain)$")
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_explanation: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    recommendation: str = Field(pattern="^(proceed|reconsider)$")
    reasoning_summary: str = ""
    evidence_citations: list[str] = Field(default_factory=list)
    suggested_action_family: str = ""
    suggested_action_allowed: bool = False
    lesson_quality: str = Field(default="adequate", pattern="^(weak|adequate|strong)$")
    panel: list[dict[str, Any]] = Field(default_factory=list)
    agreement_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    dissent_summary: str = ""


@runtime_checkable
class MemoryPort(Protocol):
    def retrieve_similar_lessons(self, query: str, k: int) -> list: ...

    def store_experiment(self, data: dict) -> str: ...

    def store_lesson(self, data: dict) -> str: ...

    def commit_experience(self, experiment: dict, lesson: dict) -> dict[str, str]: ...

    def get_run_history(self, run_id: str) -> list: ...


@runtime_checkable
class MLPort(Protocol):
    def train_model(self, config: dict) -> dict: ...

    def evaluate_model(self, model_id: str) -> dict: ...

    def apply_experiment(self, current_config: dict, proposal: dict) -> dict: ...


@runtime_checkable
class ReasoningPort(Protocol):
    def diagnose(self, state: dict, memories: list) -> dict: ...

    def propose_experiment(self, state: dict, diagnosis: dict) -> dict: ...

    def critique_result(self, before: dict, after: dict, action: dict) -> dict: ...


@runtime_checkable
class JudgePort(Protocol):
    def second_opinion(self, context: dict) -> dict: ...


def normalize_metrics(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    metrics = dict(raw.get("metrics") or {})
    if "macro_f1" not in metrics and "macro_f1" in raw:
        metrics["macro_f1"] = raw["macro_f1"]
    if "accuracy" not in metrics and "accuracy" in raw:
        metrics["accuracy"] = raw["accuracy"]
    return {
        "metrics": metrics,
        "confusion_matrix": dict(raw.get("confusion_matrix") or {}),
        "per_class_metrics": dict(raw.get("per_class_metrics") or {}),
    }


def flatten_metrics(eval_result: dict[str, Any]) -> dict[str, Any]:
    """Return a flat metric dict suitable for state.*_metrics fields."""
    normalized = normalize_metrics(eval_result)
    flat = dict(normalized["metrics"])
    return flat


def normalize_diagnosis(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    confidence = raw.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = raw.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return {
        "summary": str(raw.get("summary") or ""),
        "confidence": confidence,
        "evidence": [str(e) for e in evidence],
    }


def normalize_proposal(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    confidence = raw.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    next_action = str(raw.get("next_action") or "")
    if next_action not in ALLOWED_ACTIONS:
        raise ProposalValidationError(
            f"Invalid next_action={next_action!r}. "
            f"Allowed: {sorted(ALLOWED_ACTIONS)}"
        )
    return {
        "diagnosis": str(raw.get("diagnosis") or ""),
        "hypothesis": str(raw.get("hypothesis") or ""),
        "next_action": next_action,
        "parameters": parameters,
        "expected_effect": str(raw.get("expected_effect") or ""),
        "confidence": confidence,
    }


def normalize_critique(
    raw: dict[str, Any] | None,
    *,
    measured_helped: bool | None = None,
) -> dict[str, Any]:
    raw = raw or {}
    lesson = dict(raw.get("lesson") or {})
    helped = raw.get("helped")
    if helped is None:
        helped = bool(measured_helped) if measured_helped is not None else False
    else:
        helped = bool(helped)
        if measured_helped is not None and helped != measured_helped:
            # Measured metrics are source of truth.
            helped = measured_helped
    confidence = lesson.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "helped": helped,
        "reason": str(raw.get("reason") or ""),
        "lesson": {
            "failure_summary": str(lesson.get("failure_summary") or ""),
            "intervention": str(lesson.get("intervention") or ""),
            "result": str(lesson.get("result") or ""),
            "confidence": confidence,
        },
    }


def normalize_judge(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(raw or {})
    suggested = str(raw.get("suggested_action_family") or "")
    if suggested and suggested not in ALLOWED_ACTIONS:
        raw["suggested_action_family"] = ""
        raw["suggested_action_allowed"] = False
    elif suggested:
        raw["suggested_action_allowed"] = True
    else:
        raw["suggested_action_allowed"] = False

    quality = str(raw.get("lesson_quality") or "adequate")
    if quality not in LESSON_QUALITY_VALUES:
        raw["lesson_quality"] = "adequate"

    try:
        return JudgeOpinion.model_validate(raw).model_dump()
    except Exception:
        fallback = dict(JUDGE_FALLBACK)
        # Preserve useful risk flags from a partial/malformed payload when present.
        flags = raw.get("risk_flags")
        if isinstance(flags, list) and flags:
            fallback["risk_flags"] = [str(f) for f in flags]
        return fallback


def lesson_quality_confidence_multiplier(quality: str | None) -> float:
    """Annotate stored lesson confidence from judge lesson_quality (no metric invention)."""
    mapping = {"weak": 0.85, "adequate": 1.0, "strong": 1.05}
    return mapping.get(str(quality or "adequate"), 1.0)


def metric_value(metrics: dict[str, Any] | None, name: str) -> float:
    metrics = metrics or {}
    value = metrics.get(name)
    if value is None and "metrics" in metrics:
        value = (metrics.get("metrics") or {}).get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
