"""Shared fixtures. Everything here runs offline: no API key, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcb_agent.fireworks.config import FireworksSettings  # noqa: E402
from pcb_agent.fireworks.client import FireworksClient  # noqa: E402
from pcb_agent.fireworks.models import DEFAULT_MODELS, Role  # noqa: E402


@pytest.fixture
def settings() -> FireworksSettings:
    """Settings that never touch the environment."""
    return FireworksSettings(
        api_key="fw_test_key",
        models=dict(DEFAULT_MODELS),
        max_retries=2,
        backoff_base=0.0,
        backoff_max=0.0,
        structured_max_attempts=2,
    )


@pytest.fixture
def make_client(settings: FireworksSettings) -> Callable[..., FireworksClient]:
    """Build a FireworksClient backed by a scripted mock transport.

    Pass a handler `(httpx.Request) -> httpx.Response`, or a list of responses
    to return in order.
    """

    def _factory(handler: Any) -> FireworksClient:
        if isinstance(handler, list):
            queue = list(handler)

            def _sequenced(request: httpx.Request) -> httpx.Response:
                if not queue:
                    raise AssertionError("mock transport ran out of responses")
                return queue.pop(0)

            handler = _sequenced

        return FireworksClient(settings, transport=httpx.MockTransport(handler))

    return _factory


def chat_response(
    content: str | None = None,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
    status: int = 200,
) -> httpx.Response:
    """A /chat/completions response body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        status,
        json={
            "id": "cmpl-test",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5,
                               "total_tokens": 15},
        },
    )


def json_chat_response(payload: Any, **kwargs: Any) -> httpx.Response:
    return chat_response(json.dumps(payload), **kwargs)


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def embeddings_response(vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        },
    )


def rerank_response(scores: list[tuple[int, float]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"index": index, "relevance_score": score, "document": None}
                for index, score in scores
            ],
            "usage": {"prompt_tokens": 20, "total_tokens": 20},
        },
    )


def error_response(status: int, message: str = "boom") -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message}})


__all__ = [
    "chat_response",
    "json_chat_response",
    "tool_call",
    "embeddings_response",
    "rerank_response",
    "error_response",
    "Role",
]
