"""Error taxonomy for the Fireworks integration.

Callers should be able to tell "retry this" from "fix your request" from
"the model gave us garbage" without parsing strings, because the LangGraph
orchestrator has different recovery behaviour for each.
"""

from __future__ import annotations

from typing import Any


class FireworksError(Exception):
    """Base class for every failure raised by this package."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 payload: Any = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload
        self.request_id = request_id

    def __str__(self) -> str:  # pragma: no cover - trivial
        bits = [self.message]
        if self.status_code is not None:
            bits.append(f"status={self.status_code}")
        if self.request_id:
            bits.append(f"request_id={self.request_id}")
        return " ".join(bits)


class FireworksConfigError(FireworksError):
    """Missing API key, malformed base URL, unusable model role."""


class FireworksAuthError(FireworksError):
    """401/403. The key is missing, revoked, or scoped to another account."""


class FireworksNotFoundError(FireworksError):
    """404. Almost always a retired serverless model id."""


class FireworksRateLimitError(FireworksError):
    """429. Retryable after a backoff."""

    def __init__(self, message: str, *, retry_after: float | None = None,
                 **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class FireworksServerError(FireworksError):
    """5xx. Retryable."""


class FireworksTimeoutError(FireworksError):
    """Network or read timeout. Retryable."""


class FireworksBadRequestError(FireworksError):
    """4xx that is not auth/404/429. Not retryable without changing the request."""


class FireworksStructuredOutputError(FireworksError):
    """The response parsed as JSON but failed schema validation, or did not
    parse at all, after the configured number of repair attempts."""

    def __init__(self, message: str, *, raw: str | None = None,
                 attempts: int = 0, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.raw = raw
        self.attempts = attempts


class FireworksToolChoiceError(FireworksError):
    """The model returned no tool call, an unknown tool, or arguments that do
    not satisfy the tool's parameter schema."""


RETRYABLE = (
    FireworksRateLimitError,
    FireworksServerError,
    FireworksTimeoutError,
)


def is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, RETRYABLE)
