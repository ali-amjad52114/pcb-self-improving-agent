"""The single HTTP client every Fireworks call goes through.

Why hand-rolled on httpx instead of the OpenAI SDK: Fireworks exposes
parameters the OpenAI-typed clients do not model (`response_format` grammar
mode, `perf_metrics_in_response`, `prompt_cache_isolation_key`,
`reasoning_effort`, `return_token_ids`, and the `/rerank` endpoint, which has
no OpenAI equivalent at all). Routing chat, embeddings, and rerank through one
client also means retry, telemetry, and model-fallback behave identically
everywhere, which matters when a run has to survive a 4-hour hackathon.

The OpenAI-compatible surface is still available: see `langchain_adapters.py`
for `ChatFireworks` / LangGraph wiring.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import httpx

from .config import FireworksSettings
from .errors import (
    FireworksAuthError,
    FireworksBadRequestError,
    FireworksError,
    FireworksNotFoundError,
    FireworksRateLimitError,
    FireworksServerError,
    FireworksTimeoutError,
    is_retryable,
)
from .models import Role, alternate_id
from .telemetry import CallRecord, UsageLedger

log = logging.getLogger("pcb_agent.fireworks")

_USER_AGENT = "pcb-self-improving-agent/0.1 (+fireworks)"


class FireworksClient:
    """Synchronous client. Safe to share across threads (httpx.Client is)."""

    def __init__(
        self,
        settings: FireworksSettings | None = None,
        *,
        ledger: UsageLedger | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or FireworksSettings.from_env()
        self.ledger = ledger if ledger is not None else UsageLedger()
        self._http = httpx.Client(
            base_url=self.settings.base_url,
            timeout=httpx.Timeout(
                self.settings.timeout, connect=self.settings.connect_timeout
            ),
            headers=self._base_headers(),
            transport=transport,
        )

    # ------------------------------------------------------------------ setup

    def _base_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }
        if self.settings.prompt_cache_isolation_key:
            headers["x-prompt-cache-isolation-key"] = self.settings.prompt_cache_isolation_key
        return headers

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "FireworksClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------- chat

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        role: Role = Role.SCIENTIST,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
        response_format: Mapping[str, Any] | None = None,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        purpose: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /chat/completions and return the parsed response body."""
        model_id = model or self.settings.model_for(role)
        body = build_chat_body(
            self.settings,
            messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            response_format=response_format,
            stop=stop,
            seed=seed,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
        )
        return self._post_json(
            "/chat/completions", body, role=role, purpose=purpose or "chat"
        )

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        role: Role = Role.SCIENTIST,
        model: str | None = None,
        purpose: str | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw SSE chunks from a streaming chat completion.

        Use `stream_text` if you only want the assistant's text; this exists so
        the orchestrator can also watch tool-call arguments arrive incrementally.
        """
        model_id = model or self.settings.model_for(role)
        body = build_chat_body(self.settings, messages, model=model_id, **kwargs)
        body["stream"] = True
        body.setdefault("stream_options", {"include_usage": True})

        started = time.time()
        record = CallRecord(
            endpoint="/chat/completions",
            model=model_id,
            role=role.value,
            purpose=purpose or "chat.stream",
        )
        try:
            with self._http.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    response.read()
                    raise _map_status(response, model_id)
                record.request_id = response.headers.get("x-request-id")
                for chunk in _iter_sse(response.iter_lines()):
                    usage = chunk.get("usage")
                    if usage:
                        _apply_usage(record, usage)
                    yield chunk
        except httpx.TimeoutException as exc:
            record.ok = False
            record.error = f"timeout: {exc}"
            raise FireworksTimeoutError(f"streaming chat timed out: {exc}") from exc
        except FireworksError as exc:
            record.ok = False
            record.error = str(exc)
            raise
        finally:
            record.latency_s = round(time.time() - started, 4)
            if self.settings.track_usage:
                self.ledger.record(record)

    def stream_text(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Iterator[str]:
        """Yield only assistant text deltas. Handy for a live demo console."""
        for chunk in self.stream_chat(messages, **kwargs):
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    yield piece

    # ------------------------------------------------------------- embeddings

    def embed(
        self,
        inputs: str | Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = ...,  # type: ignore[assignment]
        purpose: str | None = None,
    ) -> list[list[float]]:
        """POST /embeddings. Returns vectors in the order of `inputs`."""
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if not texts:
            return []
        model_id = model or self.settings.model_for(Role.EMBEDDING)
        dims = self.settings.embedding_dimensions if dimensions is ... else dimensions

        body: dict[str, Any] = {"model": model_id, "input": texts}
        if dims:
            body["dimensions"] = int(dims)

        data = self._post_json(
            "/embeddings", body, role=Role.EMBEDDING, purpose=purpose or "embed"
        )
        rows = sorted(data.get("data", []), key=lambda row: row.get("index", 0))
        vectors = [row["embedding"] for row in rows]
        if len(vectors) != len(texts):
            raise FireworksError(
                f"embeddings returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    # ----------------------------------------------------------------- rerank

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str | None = None,
        top_n: int | None = None,
        return_documents: bool = True,
        task: str | None = None,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        """POST /rerank. Returns `[{index, relevance_score, document}]`, best first."""
        if not documents:
            return []
        body: dict[str, Any] = {
            "model": model or self.settings.model_for(Role.RERANK),
            "query": query,
            "documents": list(documents),
            "return_documents": return_documents,
        }
        if top_n is not None:
            body["top_n"] = int(top_n)
        if task:
            body["task"] = task

        data = self._post_json(
            "/rerank", body, role=Role.RERANK, purpose=purpose or "rerank"
        )
        return sorted(
            data.get("data", []),
            key=lambda row: row.get("relevance_score", 0.0),
            reverse=True,
        )

    # ----------------------------------------------------------------- models

    def list_models(self) -> list[str]:
        """GET /models. Used by the registry validator to spot retired ids."""
        response = self._request_with_retries("GET", "/models", None, model="-")
        payload = response.json()
        return [row["id"] for row in payload.get("data", []) if "id" in row]

    # --------------------------------------------------------------- internal

    def _post_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        role: Role | None,
        purpose: str,
    ) -> dict[str, Any]:
        model_id = str(body.get("model", "-"))
        started = time.time()
        record = CallRecord(
            endpoint=path,
            model=model_id,
            role=role.value if role else None,
            purpose=purpose,
        )
        try:
            response = self._request_with_retries("POST", path, body, model=model_id,
                                                  record=record)
            payload = response.json()
        except FireworksError as exc:
            record.ok = False
            record.error = str(exc)
            record.latency_s = round(time.time() - started, 4)
            if self.settings.track_usage:
                self.ledger.record(record)
            raise

        record.latency_s = round(time.time() - started, 4)
        record.request_id = response.headers.get("x-request-id")
        _apply_usage(record, payload.get("usage") or {})
        _apply_cache_headers(record, response.headers)
        choices = payload.get("choices") or []
        if choices:
            record.finish_reason = choices[0].get("finish_reason")
        if isinstance(payload.get("perf_metrics"), dict):
            record.perf_metrics = payload["perf_metrics"]
        if self.settings.track_usage:
            self.ledger.record(record)

        if self.settings.log_requests:
            log.info("%s %s -> %s", path, model_id, record.finish_reason or "ok")
        return payload

    def _request_with_retries(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        *,
        model: str,
        record: CallRecord | None = None,
    ) -> httpx.Response:
        attempt = 0
        tried_alternate = False
        last_exc: FireworksError | None = None

        while attempt <= self.settings.max_retries:
            attempt += 1
            if record is not None:
                record.attempts = attempt
            try:
                response = self._http.request(method, path, json=body)
                if response.status_code < 400:
                    return response
                raise _map_status(response, model)
            except httpx.TimeoutException as exc:
                last_exc = FireworksTimeoutError(f"{method} {path} timed out: {exc}")
            except httpx.HTTPError as exc:
                last_exc = FireworksServerError(f"{method} {path} transport error: {exc}")
            except FireworksNotFoundError:
                # A 404 here is nearly always a retired serverless model id.
                # Try the other accepted spelling once before surfacing it.
                alt = alternate_id(model) if body and "model" in body else None
                if alt and not tried_alternate:
                    tried_alternate = True
                    attempt -= 1
                    log.warning("model %s returned 404, retrying as %s", model, alt)
                    body["model"] = alt
                    model = alt
                    continue
                raise
            except FireworksError as exc:
                if not is_retryable(exc):
                    raise
                last_exc = exc

            if attempt > self.settings.max_retries:
                break
            delay = self._backoff(attempt, last_exc)
            log.warning(
                "fireworks %s %s failed (attempt %d/%d): %s; retrying in %.2fs",
                method, path, attempt, self.settings.max_retries + 1, last_exc, delay,
            )
            time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    def _backoff(self, attempt: int, exc: FireworksError | None) -> float:
        if isinstance(exc, FireworksRateLimitError) and exc.retry_after:
            return min(exc.retry_after, self.settings.backoff_max)
        raw = self.settings.backoff_base * (2 ** (attempt - 1))
        return min(raw, self.settings.backoff_max) * (0.5 + random.random() / 2)


# --------------------------------------------------------------------- shared


def build_chat_body(
    settings: FireworksSettings,
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = None,
    response_format: Mapping[str, Any] | None = None,
    stop: Sequence[str] | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a /chat/completions payload. Pure, so it is easy to unit test."""
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
    }
    if tools:
        body["tools"] = list(tools)
        # Fireworks defaults to "auto"; be explicit so the scientist node cannot
        # silently return prose instead of picking an experiment.
        body["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parallel_tool_calls
    elif tool_choice is not None:
        body["tool_choice"] = tool_choice
    if response_format:
        body["response_format"] = dict(response_format)
    if stop:
        body["stop"] = list(stop)
    if seed is not None:
        body["seed"] = seed

    effort = reasoning_effort if reasoning_effort is not None else settings.reasoning_effort
    if effort:
        body["reasoning_effort"] = effort
    if settings.perf_metrics:
        body["perf_metrics_in_response"] = True
    if settings.prompt_cache_isolation_key:
        body["prompt_cache_isolation_key"] = settings.prompt_cache_isolation_key
    if extra_body:
        body.update(extra_body)
    return body


def _map_status(response: httpx.Response, model: str) -> FireworksError:
    status = response.status_code
    request_id = response.headers.get("x-request-id")
    try:
        payload = response.json()
        detail = payload.get("error", payload)
        if isinstance(detail, dict):
            detail = detail.get("message") or json.dumps(detail)[:400]
    except (json.JSONDecodeError, ValueError):
        payload = None
        detail = (response.text or "")[:400]

    common = {"status_code": status, "payload": payload, "request_id": request_id}

    if status in (401, 403):
        return FireworksAuthError(
            f"Fireworks rejected the API key ({status}): {detail}", **common
        )
    if status == 404:
        return FireworksNotFoundError(
            f"Fireworks has no model {model!r} ({status}): {detail}. "
            "The serverless catalogue rotates; see FIREWORKS.md > Model registry.",
            **common,
        )
    if status == 429:
        retry_after = _parse_retry_after(response.headers.get("retry-after"))
        return FireworksRateLimitError(
            f"Fireworks rate limit hit: {detail}", retry_after=retry_after, **common
        )
    if status >= 500:
        return FireworksServerError(f"Fireworks server error {status}: {detail}", **common)
    return FireworksBadRequestError(f"Fireworks rejected the request ({status}): {detail}",
                                    **common)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _apply_usage(record: CallRecord, usage: Mapping[str, Any]) -> None:
    if not usage:
        return
    record.prompt_tokens = int(usage.get("prompt_tokens", record.prompt_tokens) or 0)
    record.completion_tokens = int(
        usage.get("completion_tokens", record.completion_tokens) or 0
    )
    record.total_tokens = int(
        usage.get("total_tokens", record.prompt_tokens + record.completion_tokens) or 0
    )
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping) and details.get("cached_tokens") is not None:
        record.cached_prompt_tokens = int(details["cached_tokens"])


def _apply_cache_headers(record: CallRecord, headers: Mapping[str, str]) -> None:
    """Dedicated deployments report cache hits in headers; serverless does not."""
    cached = headers.get("fireworks-cached-prompt-tokens")
    if cached is None:
        return
    try:
        record.cached_prompt_tokens = int(cached)
    except ValueError:
        pass


def _iter_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse an OpenAI-style `data: {...}` SSE stream into dicts."""
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            log.debug("skipping unparseable SSE frame: %r", data[:120])
