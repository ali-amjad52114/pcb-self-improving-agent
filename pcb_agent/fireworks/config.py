"""Configuration for the Fireworks layer.

Everything is read from the environment (see `.env.example`) so that the
LangGraph orchestrator, the smoke test, and the notebooks all agree on a
single source of truth. Nothing here performs network calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import FireworksConfigError
from .models import Role, resolve

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise FireworksConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise FireworksConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FireworksSettings:
    """Immutable settings bundle. Build with `FireworksSettings.from_env()`."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL

    # Model roster, resolved once so a mid-run env change cannot split a run.
    models: dict[Role, str] = field(default_factory=dict)

    # Transport
    timeout: float = 120.0
    connect_timeout: float = 10.0
    max_retries: int = 4
    backoff_base: float = 0.75
    backoff_max: float = 20.0

    # Generation defaults. Tool selection wants low temperature; the docs
    # recommend 0.0-0.3 to stop the model inventing tool arguments.
    temperature: float = 0.2
    critic_temperature: float = 0.0
    max_tokens: int = 2048

    # Structured output repair loop
    structured_max_attempts: int = 3

    # Embeddings. `dimensions` must match numDimensions on the Atlas index.
    embedding_dimensions: int | None = 1024
    embedding_batch_size: int = 64

    # Fireworks-specific knobs
    perf_metrics: bool = False
    prompt_cache_isolation_key: str | None = None
    reasoning_effort: str | None = None  # low | medium | high | max | none

    # Telemetry
    track_usage: bool = True
    log_requests: bool = False

    @classmethod
    def from_env(cls, *, require_key: bool = True, **overrides: Any) -> "FireworksSettings":
        api_key = (os.environ.get("FIREWORKS_API_KEY") or "").strip()
        if require_key and not api_key:
            raise FireworksConfigError(
                "FIREWORKS_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "then load it with python-dotenv (see FIREWORKS.md > Setup)."
            )

        base_url = (os.environ.get("FIREWORKS_BASE_URL") or DEFAULT_BASE_URL).strip()
        base_url = base_url.rstrip("/")
        if not base_url.startswith("http"):
            raise FireworksConfigError(f"FIREWORKS_BASE_URL looks wrong: {base_url!r}")

        settings = cls(
            api_key=api_key,
            base_url=base_url,
            models={role: resolve(role) for role in Role},
            timeout=_env_float("FIREWORKS_TIMEOUT", 120.0),
            connect_timeout=_env_float("FIREWORKS_CONNECT_TIMEOUT", 10.0),
            max_retries=_env_int("FIREWORKS_MAX_RETRIES", 4),
            backoff_base=_env_float("FIREWORKS_BACKOFF_BASE", 0.75),
            backoff_max=_env_float("FIREWORKS_BACKOFF_MAX", 20.0),
            temperature=_env_float("FIREWORKS_TEMPERATURE", 0.2),
            critic_temperature=_env_float("FIREWORKS_CRITIC_TEMPERATURE", 0.0),
            max_tokens=_env_int("FIREWORKS_MAX_TOKENS", 2048),
            structured_max_attempts=_env_int("FIREWORKS_STRUCTURED_MAX_ATTEMPTS", 3),
            embedding_dimensions=_optional_int("FIREWORKS_EMBEDDING_DIMENSIONS", 1024),
            embedding_batch_size=_env_int("FIREWORKS_EMBEDDING_BATCH_SIZE", 64),
            perf_metrics=_env_bool("FIREWORKS_PERF_METRICS", False),
            prompt_cache_isolation_key=_optional_str("FIREWORKS_PROMPT_CACHE_ISOLATION_KEY"),
            reasoning_effort=_optional_str("FIREWORKS_REASONING_EFFORT"),
            track_usage=_env_bool("FIREWORKS_TRACK_USAGE", True),
            log_requests=_env_bool("FIREWORKS_LOG_REQUESTS", False),
        )
        if overrides:
            settings = replace(settings, **overrides)
        return settings

    def model_for(self, role: Role) -> str:
        try:
            return self.models[role]
        except KeyError as exc:  # pragma: no cover - only if models dict is hand-built
            raise FireworksConfigError(f"No model configured for role {role.value!r}") from exc

    def with_model(self, role: Role, model_id: str) -> "FireworksSettings":
        """Return a copy with one role remapped. Used by registry repair."""
        models = dict(self.models)
        models[role] = model_id
        return replace(self, models=models)

    def redacted(self) -> dict[str, Any]:
        """Safe to log or dump into an experiment record."""
        return {
            "base_url": self.base_url,
            "api_key": _mask(self.api_key),
            "models": {role.value: mid for role, mid in sorted(
                self.models.items(), key=lambda kv: kv[0].value)},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "embedding_dimensions": self.embedding_dimensions,
            "reasoning_effort": self.reasoning_effort,
        }


def _optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else None


def _optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    if raw.strip().lower() in {"none", "native", "full"}:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise FireworksConfigError(f"{name} must be an integer or 'none', got {raw!r}") from exc


def _mask(secret: str) -> str:
    if not secret:
        return "<unset>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"
