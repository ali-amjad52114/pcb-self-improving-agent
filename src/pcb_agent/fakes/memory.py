"""Deterministic in-memory (optional file) experience store.

[FALLBACK ONLY — FINAL DEMO SHOULD USE MONGODB]
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


class FakeMemory:
    """Implements Person 1 memory contract with deterministic retrieval."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._experiments: list[dict[str, Any]] = []
        self._lessons: list[dict[str, Any]] = []
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def retrieve_similar_lessons(self, query: str, k: int) -> list:
        query_l = (query or "").lower()
        scored: list[tuple[float, dict[str, Any]]] = []
        for lesson in self._lessons:
            text = " ".join(
                str(lesson.get(field, ""))
                for field in (
                    "failure_summary",
                    "intervention",
                    "result",
                    "next_action",
                )
            ).lower()
            score = 0.0
            for token in query_l.replace(",", " ").split():
                if len(token) < 3:
                    continue
                if token in text:
                    score += 1.0
            # Prefer lessons that mention minority/open-circuit when query does.
            if "open" in query_l and "open" in text:
                score += 2.0
            if "minority" in query_l and "minority" in text:
                score += 2.0
            if "sampler" in text and ("recall" in query_l or "minority" in query_l):
                score += 1.5
            scored.append((score, lesson))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [item[1] for item in scored if item[0] > 0][:k]
        if not results and self._lessons:
            # Soft fallback: return most recent lessons so warm runs still see memory.
            results = list(reversed(self._lessons))[:k]
        return [dict(r) for r in results]

    def store_experiment(self, data: dict) -> str:
        record = dict(data)
        experiment_id = str(record.get("experiment_id") or f"exp_{uuid.uuid4().hex[:8]}")
        record["experiment_id"] = experiment_id
        self._experiments.append(record)
        self._save()
        return experiment_id

    def store_lesson(self, data: dict) -> str:
        record = dict(data)
        lesson_id = str(record.get("lesson_id") or f"lesson_{len(self._lessons) + 1}")
        record["lesson_id"] = lesson_id
        self._lessons.append(record)
        self._save()
        return lesson_id

    def get_run_history(self, run_id: str) -> list:
        return [dict(e) for e in self._experiments if e.get("run_id") == run_id]

    @property
    def lessons(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._lessons]

    @property
    def experiments(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._experiments]

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"experiments": self._experiments, "lessons": self._lessons}
        self._persist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        assert self._persist_path is not None
        payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        self._experiments = list(payload.get("experiments") or [])
        self._lessons = list(payload.get("lessons") or [])
