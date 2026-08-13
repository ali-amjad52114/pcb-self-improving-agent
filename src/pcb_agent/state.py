"""LangGraph agent state — JSON-serializable only."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    run_id: str

    iteration: int
    experiment_budget: int

    target_metric: str
    target_value: float

    dataset_summary: dict[str, Any]

    current_config: dict[str, Any]
    best_config: dict[str, Any]

    current_model_id: str

    baseline_metrics: dict[str, Any]
    previous_metrics: dict[str, Any]
    current_metrics: dict[str, Any]
    best_metrics: dict[str, Any]

    confusion_matrix: dict[str, Any]
    per_class_metrics: dict[str, Any]

    memory_query: str
    retrieved_lessons: list[dict[str, Any]]

    diagnosis: dict[str, Any]
    proposed_experiment: dict[str, Any]

    proposal_revision_count: int

    evaluator_opinion: dict[str, Any]
    evaluator_trigger: str

    critique: dict[str, Any]

    experiment_history: list[dict[str, Any]]

    consecutive_failures: int

    last_metric_delta: float
    last_experiment_helped: bool

    status: str
    stop_reason: str

    warnings: list[str]

    # Orchestrator bookkeeping (still JSON-safe)
    pending_experiment_id: str
    openrouter_calls: int
    openrouter_call_budget: int
    openrouter_ledger: list[dict[str, Any]]
    judge_skipped: bool
    judge_skip_reason: str
