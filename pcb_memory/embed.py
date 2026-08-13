"""Voyage embeddings for Atlas Vector Search."""

from __future__ import annotations

import os

import voyageai
from dotenv import load_dotenv

from pcb_memory.schema import EMBED_MODEL

load_dotenv()

_client: voyageai.Client | None = None


def voyage_client() -> voyageai.Client:
    global _client
    if _client is None:
        key = os.environ.get("VOYAGE_API_KEY") or os.environ.get("VOYAGEAI_API_KEY")
        if not key:
            raise RuntimeError("Set VOYAGE_API_KEY in .env")
        _client = voyageai.Client(api_key=key)
    return _client


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    result = voyage_client().embed(texts, model=EMBED_MODEL, input_type=input_type)
    return result.embeddings


def embed_query(text: str) -> list[float]:
    return embed_texts([text], input_type="query")[0]


def embed_document(text: str) -> list[float]:
    return embed_texts([text], input_type="document")[0]
