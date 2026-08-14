"""Fireworks embeddings for MongoDB Atlas Vector Search.

The scaffold shipped with Voyage placeholders. Fireworks also serves Voyage
models, plus Qwen3 embeddings, through the same key and the same base URL,
which means the memory layer can run on the Fireworks credit instead of a
second vendor and a second key. `qwen3-embedding-8b` supports Matryoshka
truncation via `dimensions`, so you can index at 1024 dims and keep Atlas
index size and query latency down.

Hard rule: the embedding model and `dimensions` are part of the index. If you
change either, every vector already in Atlas is meaningless and the index must
be rebuilt. That is why `Role.EMBEDDING` is in `models.PINNED_ROLES` and never
auto-falls-back.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Iterator, Mapping, Sequence

from .client import FireworksClient
from .errors import FireworksConfigError
from .models import Role

log = logging.getLogger("pcb_agent.fireworks.embeddings")


class FireworksEmbedder:
    """Batching, caching wrapper over `/embeddings`.

    The cache is per-instance and keyed on (model, dimensions, text). Lesson
    text is re-embedded constantly during a run (same failure signature queried
    each iteration), so this removes a meaningful number of calls.
    """

    def __init__(
        self,
        client: FireworksClient,
        *,
        model: str | None = None,
        dimensions: int | None = ...,  # type: ignore[assignment]
        batch_size: int | None = None,
        cache: bool = True,
    ) -> None:
        self.client = client
        self.model = model or client.settings.model_for(Role.EMBEDDING)
        self.dimensions = (
            client.settings.embedding_dimensions if dimensions is ... else dimensions
        )
        self.batch_size = batch_size or client.settings.embedding_batch_size
        self._cache: dict[str, list[float]] | None = {} if cache else None

    # ------------------------------------------------------------------ core

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed lesson/experiment text for storage. Order is preserved."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a failure summary to search Atlas with."""
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]| None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []

        for index, text in enumerate(texts):
            key = self._key(text)
            if self._cache is not None and key in self._cache:
                results[index] = self._cache[key]
            else:
                pending.append((index, text))

        for chunk in _chunks(pending, self.batch_size):
            vectors = self.client.embed(
                [text for _, text in chunk],
                model=self.model,
                dimensions=self.dimensions,
                purpose="embed_documents",
            )
            for (index, text), vector in zip(chunk, vectors):
                results[index] = vector
                if self._cache is not None:
                    self._cache[self._key(text)] = vector

        missing = [i for i, v in enumerate(results) if v is None]
        if missing:  # pragma: no cover - client.embed already length-checks
            raise FireworksConfigError(f"embedding failed for input indices {missing}")
        return [v for v in results if v is not None]

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.model}:{self.dimensions}:{digest}"

    # ---------------------------------------------------------------- helpers

    @property
    def vector_size(self) -> int:
        """Dimensionality of the vectors this embedder produces.

        Probes the API once when `dimensions` was left native, because the
        Atlas index needs an exact number.
        """
        if self.dimensions:
            return int(self.dimensions)
        return len(self.embed_query("probe"))

    def atlas_index_definition(
        self, *, path: str = "embedding", similarity: str = "cosine",
        filters: Sequence[str] = ("failure_mode", "reusable", "avoid"),
    ) -> dict[str, Any]:
        """The Atlas Vector Search index definition matching this embedder.

        Hand this to Person 1 (or `db.collection.createSearchIndex`) so the
        index and the vectors cannot drift apart.
        """
        fields: list[dict[str, Any]] = [
            {
                "type": "vector",
                "path": path,
                "numDimensions": self.vector_size,
                "similarity": similarity,
            }
        ]
        fields.extend({"type": "filter", "path": name} for name in filters)
        return {"fields": fields}

    def describe(self) -> dict[str, Any]:
        """Provenance stamped onto every stored vector, so a later model change
        is detectable instead of silently corrupting retrieval."""
        return {
            "embedding_model": self.model,
            "embedding_dimensions": self.dimensions,
            "embedding_provider": "fireworks",
        }


class LangChainFireworksEmbeddings:
    """Adapter exposing the LangChain `Embeddings` interface.

    Lets `MongoDBAtlasVectorSearch` and the rest of the LangChain stack use this
    client (with its retries, telemetry, and cache) rather than opening a second
    unmonitored connection to Fireworks.

        from langchain_mongodb import MongoDBAtlasVectorSearch
        store = MongoDBAtlasVectorSearch(
            collection=db["lessons"],
            embedding=LangChainFireworksEmbeddings(embedder),
            index_name="lessons_vector_index",
        )
    """

    def __init__(self, embedder: FireworksEmbedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Local similarity, for tests and for ranking without a round trip."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def attach_embeddings(
    embedder: FireworksEmbedder,
    documents: Sequence[Mapping[str, Any]],
    *,
    text_field: str = "embedding_text",
    vector_field: str = "embedding",
) -> list[dict[str, Any]]:
    """Embed a batch of lesson documents in one pass and attach the vectors.

    Skips documents that already carry a vector, so re-running the writer after
    a crash does not re-embed everything.
    """
    out = [dict(doc) for doc in documents]
    todo = [
        (index, str(doc.get(text_field) or ""))
        for index, doc in enumerate(out)
        if not doc.get(vector_field) and doc.get(text_field)
    ]
    if not todo:
        return out

    vectors = embedder.embed_documents([text for _, text in todo])
    provenance = embedder.describe()
    for (index, _), vector in zip(todo, vectors):
        out[index][vector_field] = vector
        out[index].update(provenance)
    return out


def _chunks(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), max(1, size)):
        yield list(items[start:start + size])
