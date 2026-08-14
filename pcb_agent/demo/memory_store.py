"""An in-process stand-in for the MongoDB memory layer.

Person 1 owns the real Atlas implementation. This exists so the Fireworks side
can be demonstrated and tested without a cluster, and it deliberately exposes
the exact frozen interface from the 3-person plan:

    store_experiment(data) -> str
    store_lesson(data) -> str
    retrieve_similar_lessons(query, k) -> list
    get_run_history(run_id) -> list

Retrieval uses real Fireworks embeddings and brute-force cosine similarity.
That is genuinely what `$vectorSearch` computes; Atlas just does it with an
HNSW index over a collection too large to scan. At hackathon scale (tens of
lessons) the results are identical, so swapping this for the Atlas client is a
drop-in change rather than a behavioural one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from pcb_agent.fireworks.embeddings import FireworksEmbedder, cosine_similarity

log = logging.getLogger("pcb_agent.demo.memory")


class InMemoryLessonStore:
    """JSON-backed lesson and experiment store with vector retrieval."""

    def __init__(self, embedder: FireworksEmbedder, path: str | Path | None = None) -> None:
        self.embedder = embedder
        self.path = Path(path) if path else None
        self.lessons: list[dict[str, Any]] = []
        self.experiments: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            self.load()

    # ------------------------------------------------------------ frozen API

    def store_experiment(self, data: Mapping[str, Any]) -> str:
        record = dict(data)
        record.setdefault("experiment_id", f"exp_{len(self.experiments) + 1:03d}")
        self.experiments.append(record)
        self._flush()
        return str(record["experiment_id"])

    def store_lesson(self, data: Mapping[str, Any]) -> str:
        record = dict(data)
        record.setdefault("lesson_id", f"lesson_{len(self.lessons) + 1:03d}")
        text = record.get("embedding_text") or record.get("failure_summary") or ""
        if text and not record.get("embedding"):
            record["embedding"] = self.embedder.embed_query(text)
            record.update(self.embedder.describe())
        self.lessons.append(record)
        self._flush()
        return str(record["lesson_id"])

    def retrieve_similar_lessons(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Vector-search stage. Returns candidates with a `score`, best first.

        Only `reusable` lessons are retrievable, matching the filter the Atlas
        index is defined with.
        """
        pool = [
            lesson for lesson in self.lessons
            if lesson.get("embedding") and lesson.get("reusable", True)
        ]
        if not pool or not query.strip():
            return []

        query_vector = self.embedder.embed_query(query)
        scored = [
            {**lesson, "score": cosine_similarity(query_vector, lesson["embedding"])}
            for lesson in pool
        ]
        scored.sort(key=lambda row: row["score"], reverse=True)
        # Drop the raw vector before it reaches a prompt; it is thousands of
        # tokens of nothing.
        return [{k2: v for k2, v in row.items() if k2 != "embedding"}
                for row in scored[:k]]

    def get_run_history(self, run_id: str) -> list[dict[str, Any]]:
        return [e for e in self.experiments if e.get("run_id") == run_id]

    # -------------------------------------------------------------- storage

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("lesson store at %s is corrupt; starting empty", self.path)
            return
        self.lessons = payload.get("lessons", [])
        self.experiments = payload.get("experiments", [])
        log.info("loaded %d lessons, %d experiments from %s",
                 len(self.lessons), len(self.experiments), self.path)

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"lessons": self.lessons, "experiments": self.experiments},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.lessons.clear()
        self.experiments.clear()
        self._flush()

    def stats(self) -> dict[str, Any]:
        return {
            "lessons": len(self.lessons),
            "reusable_lessons": sum(1 for l in self.lessons if l.get("reusable", True)),
            "avoid_lessons": sum(1 for l in self.lessons if l.get("avoid")),
            "experiments": len(self.experiments),
            "runs": sorted({str(e.get("run_id")) for e in self.experiments if e.get("run_id")}),
        }


def to_atlas_documents(lessons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Hand the demo's lessons to Person 1 for a real Atlas insert_many."""
    return [dict(lesson) for lesson in lessons]
