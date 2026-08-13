"""Environment-backed settings via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    agent_mode: Literal["fake", "integrated"] = Field(default="fake", alias="AGENT_MODE")

    target_metric: str = Field(default="macro_f1", alias="TARGET_METRIC")
    target_value: float = Field(default=0.84, alias="TARGET_VALUE")
    experiment_budget: int = Field(default=4, alias="EXPERIMENT_BUDGET")

    memory_module: str = Field(default="", alias="MEMORY_MODULE")
    ml_module: str = Field(default="", alias="ML_MODULE")
    reasoning_module: str = Field(default="", alias="REASONING_MODULE")

    checkpointer_backend: Literal["memory", "mongodb"] = Field(
        default="memory", alias="CHECKPOINTER_BACKEND"
    )
    mongodb_uri: str = Field(default="", alias="MONGODB_URI")
    mongodb_db_name: str = Field(
        default="persistent_context", alias="MONGODB_DB_NAME"
    )

    openrouter_enabled: bool = Field(default=False, alias="OPENROUTER_ENABLED")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(default="", alias="OPENROUTER_MODEL")
    openrouter_model_secondary: str = Field(
        default="", alias="OPENROUTER_MODEL_SECONDARY"
    )
    openrouter_model_tertiary: str = Field(
        default="", alias="OPENROUTER_MODEL_TERTIARY"
    )
    openrouter_provider_order: str = Field(
        default="", alias="OPENROUTER_PROVIDER_ORDER"
    )
    openrouter_timeout_seconds: float = Field(
        default=20.0, alias="OPENROUTER_TIMEOUT_SECONDS"
    )
    openrouter_max_retries: int = Field(default=2, alias="OPENROUTER_MAX_RETRIES")
    openrouter_call_budget: int = Field(default=5, alias="OPENROUTER_CALL_BUDGET")
    openrouter_max_tokens: int = Field(default=800, alias="OPENROUTER_MAX_TOKENS")

    memory_top_k: int = Field(default=5, alias="MEMORY_TOP_K")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def openrouter_models(self) -> list[str]:
        models: list[str] = []
        for name in (
            self.openrouter_model,
            self.openrouter_model_secondary,
            self.openrouter_model_tertiary,
        ):
            cleaned = (name or "").strip()
            if cleaned and cleaned not in models:
                models.append(cleaned)
        return models

    def openrouter_provider_list(self) -> list[str]:
        raw = (self.openrouter_provider_order or "").strip()
        if not raw:
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
