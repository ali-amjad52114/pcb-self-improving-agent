"""Deterministic memory search query — no LLM."""

from __future__ import annotations

from typing import Any

from pcb_agent.contracts import metric_value
from pcb_agent.state import AgentState


def build_memory_query(state: AgentState) -> str:
    parts: list[str] = []

    per_class = state.get("per_class_metrics") or {}
    for class_name, metrics in per_class.items():
        if not isinstance(metrics, dict):
            continue
        recall = metrics.get("recall")
        try:
            recall_f = float(recall)
        except (TypeError, ValueError):
            continue
        if recall_f < 0.6:
            parts.append(f"{class_name} defects have low recall")
            if class_name in {"open_circuit", "open-circuit"}:
                parts.append("Open-circuit is a minority class")

    dataset = state.get("dataset_summary") or {}
    imbalance = dataset.get("class_imbalance_ratio") or dataset.get("imbalance_ratio")
    if imbalance:
        parts.append(f"class imbalance ratio {imbalance}")
    elif any("minority" in p.lower() or "open-circuit" in p.lower() or "open_circuit" in p.lower() for p in parts):
        parts.append("class imbalance")

    target_metric = state.get("target_metric") or "macro_f1"
    current = metric_value(state.get("current_metrics"), target_metric)
    if current:
        parts.append(f"current {target_metric} {current:.3f}")

    history = state.get("experiment_history") or []
    for item in history[-3:]:
        if item.get("helped"):
            continue
        proposal = item.get("proposal") or {}
        action = proposal.get("next_action")
        if action:
            parts.append(
                f"Previous {action} experiment did not improve {target_metric}"
            )

    if not parts:
        parts.append("PCB defect classification minority class recall failure")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(part)

    return ". ".join(unique) + ("." if unique else "")


def summarize_lessons_for_console(lessons: list[dict[str, Any]]) -> list[tuple[bool, str]]:
    """Return (helped, summary) pairs for demo REUSED EXPERIENCE lines."""
    rows: list[tuple[bool, str]] = []
    for lesson in lessons:
        helped = bool(lesson.get("helped", True))
        intervention = str(lesson.get("intervention") or lesson.get("next_action") or "intervention")
        failure = str(lesson.get("failure_summary") or "similar failure")
        if helped:
            rows.append((True, f"{intervention} improved similar {failure}"))
        else:
            rows.append((False, f"{intervention} previously failed"))
    return rows
