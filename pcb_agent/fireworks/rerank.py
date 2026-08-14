"""Fireworks reranking as a precision pass over Atlas Vector Search recall.

Vector search on a small lesson collection has a characteristic failure: with
20 lessons stored, cosine similarity happily returns the 5 "closest" even when
none of them is actually about the current failure. The scientist then reads
five irrelevant lessons and either ignores them or, worse, follows one.

The fix is the standard two-stage retrieval shape:

    Atlas Vector Search  ->  top 20 by cosine   (recall, cheap, approximate)
    Fireworks /rerank    ->  top 5 by relevance (precision, cross-encoder)
    relevance floor      ->  possibly 0 lessons (honest cold start)

The floor matters more than the ordering. An agent that correctly reports "no
relevant prior experience" is what makes the Run A vs Run B comparison
credible; an agent that always returns 5 lessons has no cold start to contrast
against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .client import FireworksClient

log = logging.getLogger("pcb_agent.fireworks.rerank")

#: Below this cross-encoder score a lesson is treated as unrelated. Tuned to be
#: deliberately strict: a false "relevant" lesson costs a whole training run.
DEFAULT_RELEVANCE_FLOOR = 0.35

RERANK_TASK = (
    "Given a machine learning failure description, rank prior lessons by how "
    "directly they would help decide the next experiment. A lesson is relevant "
    "only if it addresses the same kind of failure, not merely the same domain."
)


@dataclass
class RankedLesson:
    """A retrieved lesson with both retrieval scores kept separate."""

    lesson: dict[str, Any]
    vector_score: float | None
    rerank_score: float | None
    rank: int

    def to_memory(self) -> dict[str, Any]:
        """Shape consumed by `scientist.diagnose(..., memories=...)`."""
        memory = dict(self.lesson)
        memory["score"] = self.rerank_score if self.rerank_score is not None else self.vector_score
        memory["vector_score"] = self.vector_score
        memory["rerank_score"] = self.rerank_score
        memory["rank"] = self.rank
        return memory


def rerank_lessons(
    client: FireworksClient,
    failure_summary: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
    relevance_floor: float = DEFAULT_RELEVANCE_FLOOR,
    text_field: str = "embedding_text",
    fallback_field: str = "failure_summary",
) -> list[RankedLesson]:
    """Rerank Atlas candidates against the current failure.

    `candidates` are the raw documents from `retrieve_similar_lessons`, each
    optionally carrying a `score` from the vector stage.
    """
    if not candidates:
        return []

    documents = [
        str(doc.get(text_field) or doc.get(fallback_field) or doc.get("lesson") or "")
        for doc in candidates
    ]
    keep = [i for i, text in enumerate(documents) if text.strip()]
    if not keep:
        log.warning("all %d candidates had empty text; skipping rerank", len(candidates))
        return []

    rows = client.rerank(
        failure_summary,
        [documents[i] for i in keep],
        top_n=min(top_k * 2, len(keep)),
        return_documents=False,
        task=RERANK_TASK,
        purpose="rerank_lessons",
    )

    ranked: list[RankedLesson] = []
    for position, row in enumerate(rows):
        original = keep[int(row.get("index", 0))]
        doc = candidates[original]
        score = float(row.get("relevance_score", 0.0))
        if score < relevance_floor:
            continue
        ranked.append(
            RankedLesson(
                lesson=dict(doc),
                vector_score=_as_float(doc.get("score")),
                rerank_score=score,
                rank=position,
            )
        )
        if len(ranked) >= top_k:
            break

    log.info(
        "rerank: %d candidates -> %d above floor %.2f",
        len(candidates), len(ranked), relevance_floor,
    )
    return ranked


def retrieve_memories(
    client: FireworksClient,
    failure_summary: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
    relevance_floor: float = DEFAULT_RELEVANCE_FLOOR,
    use_rerank: bool = True,
) -> list[dict[str, Any]]:
    """End-to-end second stage, returning memories ready for the scientist.

    With `use_rerank=False` this degrades to "trust the vector scores", which
    is the right behaviour if the rerank model is unavailable: fewer, worse
    memories beats a crashed run.
    """
    if not candidates:
        return []

    if not use_rerank:
        top = sorted(
            candidates, key=lambda d: _as_float(d.get("score")) or 0.0, reverse=True
        )[:top_k]
        return [
            RankedLesson(dict(d), _as_float(d.get("score")), None, i).to_memory()
            for i, d in enumerate(top)
        ]

    try:
        ranked = rerank_lessons(
            client,
            failure_summary,
            candidates,
            top_k=top_k,
            relevance_floor=relevance_floor,
        )
    except Exception as exc:  # noqa: BLE001 - retrieval must never kill a run
        log.warning("rerank unavailable (%s); falling back to vector order", exc)
        return retrieve_memories(
            client, failure_summary, candidates, top_k=top_k, use_rerank=False
        )

    return [item.to_memory() for item in ranked]


def split_by_polarity(
    memories: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate "do this" lessons from "avoid this" lessons.

    The scientist prompt renders them differently, and the avoid list is what
    `tools.ActionHistory` uses to withdraw tools from the allowlist.
    """
    use = [dict(m) for m in memories if not m.get("avoid")]
    avoid = [dict(m) for m in memories if m.get("avoid")]
    return use, avoid


def actions_to_withdraw(
    memories: Sequence[Mapping[str, Any]],
    failure_mode: str | None,
    *,
    min_confidence: float = 0.6,
) -> list[str]:
    """Action names that retrieved memory says failed on this failure mode.

    Only high-confidence, same-failure-mode negatives are honoured. A lesson
    from a different failure mode is context, not a veto.
    """
    withdraw: set[str] = set()
    for memory in memories:
        if not memory.get("avoid"):
            continue
        if failure_mode and memory.get("failure_mode") not in (None, failure_mode):
            continue
        if float(memory.get("confidence", 0.0)) < min_confidence:
            continue
        action = memory.get("action") or memory.get("next_action")
        if action:
            withdraw.add(str(action))
    return sorted(withdraw)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
