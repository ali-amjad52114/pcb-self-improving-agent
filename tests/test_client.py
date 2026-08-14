"""Transport behaviour: request assembly, error mapping, retries, telemetry."""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import chat_response, embeddings_response, error_response, rerank_response

from pcb_agent.fireworks.client import build_chat_body
from pcb_agent.fireworks.config import FireworksSettings
from pcb_agent.fireworks.errors import (
    FireworksAuthError,
    FireworksBadRequestError,
    FireworksNotFoundError,
    FireworksRateLimitError,
    FireworksServerError,
)
from pcb_agent.fireworks.models import Role


class TestBuildChatBody:
    def test_defaults_come_from_settings(self, settings: FireworksSettings):
        body = build_chat_body(settings, [{"role": "user", "content": "hi"}],
                               model="m")
        assert body["model"] == "m"
        assert body["temperature"] == settings.temperature
        assert body["max_tokens"] == settings.max_tokens
        assert "tools" not in body
        assert "response_format" not in body

    def test_tools_force_an_explicit_tool_choice(self, settings: FireworksSettings):
        body = build_chat_body(settings, [], model="m",
                               tools=[{"type": "function", "function": {"name": "t"}}])
        assert body["tool_choice"] == "auto"

    def test_tool_choice_is_respected(self, settings: FireworksSettings):
        body = build_chat_body(settings, [], model="m",
                               tools=[{"type": "function", "function": {"name": "t"}}],
                               tool_choice="required", parallel_tool_calls=False)
        assert body["tool_choice"] == "required"
        assert body["parallel_tool_calls"] is False

    def test_fireworks_specific_flags(self):
        settings = FireworksSettings(
            api_key="k", perf_metrics=True, reasoning_effort="high",
            prompt_cache_isolation_key="run_a",
        )
        body = build_chat_body(settings, [], model="m")
        assert body["perf_metrics_in_response"] is True
        assert body["reasoning_effort"] == "high"
        assert body["prompt_cache_isolation_key"] == "run_a"

    def test_per_call_reasoning_effort_overrides_settings(self):
        settings = FireworksSettings(api_key="k", reasoning_effort="high")
        body = build_chat_body(settings, [], model="m", reasoning_effort="none")
        assert body["reasoning_effort"] == "none"


class TestErrorMapping:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, FireworksAuthError),
            (403, FireworksAuthError),
            (404, FireworksNotFoundError),
            (422, FireworksBadRequestError),
        ],
    )
    def test_non_retryable_statuses(self, make_client, status, expected):
        client = make_client(lambda request: error_response(status))
        with pytest.raises(expected):
            client.chat([{"role": "user", "content": "hi"}])

    def test_404_message_points_at_the_model_registry(self, make_client):
        client = make_client(lambda request: error_response(404, "model not found"))
        with pytest.raises(FireworksNotFoundError, match="catalogue rotates"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_rate_limit_is_retried_then_surfaced(self, make_client):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(429, json={"error": {"message": "slow down"}},
                                  headers={"retry-after": "0"})

        client = make_client(handler)
        with pytest.raises(FireworksRateLimitError):
            client.chat([{"role": "user", "content": "hi"}])
        # max_retries=2 in the fixture, so 3 attempts total.
        assert len(calls) == 3

    def test_server_error_recovers_on_retry(self, make_client):
        client = make_client([error_response(503), chat_response("pong")])
        payload = client.chat([{"role": "user", "content": "hi"}])
        assert payload["choices"][0]["message"]["content"] == "pong"

    def test_persistent_server_error_raises(self, make_client):
        client = make_client(lambda request: error_response(500))
        with pytest.raises(FireworksServerError):
            client.chat([{"role": "user", "content": "hi"}])


def test_404_retries_with_the_alternate_model_spelling(make_client):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model.startswith("accounts/"):
            return error_response(404, "no such model")
        return embeddings_response([[0.1, 0.2]])

    client = make_client(handler)
    vectors = client.embed(["hello"])
    assert vectors == [[0.1, 0.2]]
    assert seen[0].startswith("accounts/fireworks/models/")
    assert seen[1].startswith("fireworks/")


class TestEmbeddings:
    def test_vectors_are_returned_in_input_order(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "data": [
                    {"index": 1, "embedding": [1.0]},
                    {"index": 0, "embedding": [0.0]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            })

        client = make_client(handler)
        assert client.embed(["a", "b"]) == [[0.0], [1.0]]

    def test_dimensions_are_sent(self, make_client):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return embeddings_response([[0.0] * 8])

        client = make_client(handler)
        client.embed(["x"], dimensions=8)
        assert captured["dimensions"] == 8

    def test_empty_input_makes_no_request(self, make_client):
        client = make_client(lambda request: pytest.fail("should not be called"))
        assert client.embed([]) == []


class TestRerank:
    def test_rows_are_sorted_by_relevance(self, make_client):
        client = make_client([rerank_response([(0, 0.2), (1, 0.9), (2, 0.5)])])
        rows = client.rerank("q", ["a", "b", "c"])
        assert [r["index"] for r in rows] == [1, 2, 0]

    def test_empty_documents_makes_no_request(self, make_client):
        client = make_client(lambda request: pytest.fail("should not be called"))
        assert client.rerank("q", []) == []


class TestTelemetry:
    def test_usage_is_recorded(self, make_client):
        client = make_client([
            chat_response("a", usage={"prompt_tokens": 100, "completion_tokens": 20,
                                      "total_tokens": 120}),
            chat_response("b", usage={"prompt_tokens": 5, "completion_tokens": 3,
                                      "total_tokens": 8}),
        ])
        client.chat([{"role": "user", "content": "1"}], purpose="diagnose")
        client.chat([{"role": "user", "content": "2"}], purpose="critique")

        totals = client.ledger.totals()
        assert totals["calls"] == 2
        assert totals["prompt_tokens"] == 105
        assert totals["completion_tokens"] == 23
        assert totals["total_tokens"] == 128
        assert totals["failed_calls"] == 0
        assert set(client.ledger.by_purpose()) == {"diagnose", "critique"}

    def test_failed_calls_are_recorded(self, make_client):
        client = make_client(lambda request: error_response(401))
        with pytest.raises(FireworksAuthError):
            client.chat([{"role": "user", "content": "hi"}])
        totals = client.ledger.totals()
        assert totals["calls"] == 1
        assert totals["failed_calls"] == 1

    def test_cost_is_none_without_a_price_table(self, make_client):
        client = make_client([chat_response("x")])
        client.chat([{"role": "user", "content": "hi"}])
        assert client.ledger.totals()["cost_usd"] is None

    def test_cost_is_computed_with_a_price_table(self, make_client, settings):
        client = make_client([chat_response(
            "x", usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                        "total_tokens": 2_000_000})])
        client.ledger.prices = {
            settings.model_for(Role.SCIENTIST): {"input": 0.6, "output": 2.5}
        }
        client.chat([{"role": "user", "content": "hi"}])
        assert client.ledger.totals()["cost_usd"] == pytest.approx(3.1)


class TestStreaming:
    def test_text_deltas_are_yielded(self, make_client):
        frames = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":3,'
            '"completion_tokens":2,"total_tokens":5}}',
            "data: [DONE]",
        ]
        body = ("\n".join(frames) + "\n").encode()
        client = make_client(lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}))

        assert "".join(client.stream_text([{"role": "user", "content": "hi"}])) == \
            "Hello world"
        assert client.ledger.totals()["total_tokens"] == 5

    def test_malformed_frames_are_skipped(self, make_client):
        body = (
            "data: not-json\n"
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
            "data: [DONE]\n"
        ).encode()
        client = make_client(lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}))
        assert "".join(client.stream_text([{"role": "user", "content": "hi"}])) == "ok"


def test_list_models(make_client):
    client = make_client(lambda request: httpx.Response(
        200, json={"data": [{"id": "accounts/fireworks/models/a"},
                            {"id": "accounts/fireworks/models/b"}]}))
    assert client.list_models() == ["accounts/fireworks/models/a",
                                    "accounts/fireworks/models/b"]


def test_authorization_header_is_set(make_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return chat_response("ok")

    client = make_client(handler)
    client.chat([{"role": "user", "content": "hi"}])
    assert captured["auth"] == "Bearer fw_test_key"
