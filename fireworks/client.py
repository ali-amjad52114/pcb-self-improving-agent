"""Small Fireworks chat-completions client with structured output validation."""

from __future__ import annotations

import json
import os
import time
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class FireworksError(RuntimeError):
    """A Fireworks request or response could not satisfy the module contract."""


class FireworksClient:
    """Call Fireworks through its OpenAI-compatible structured-output API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int = 2,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("FIREWORKS_BASE_URL")
            or "https://api.fireworks.ai/inference/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("FIREWORKS_MODEL")
            or "accounts/fireworks/models/gpt-oss-20b"
        )
        self.timeout_seconds = float(
            timeout_seconds or os.getenv("FIREWORKS_TIMEOUT_SECONDS", "30")
        )
        self.max_retries = max(0, max_retries)
        if not self.api_key:
            raise FireworksError("FIREWORKS_API_KEY is not configured")

    def generate(
        self,
        *,
        system_prompt: str,
        payload: dict,
        response_model: type[T],
        schema_name: str,
    ) -> T:
        """Generate one response and validate it against a Pydantic model."""
        schema = response_model.model_json_schema()
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Analyze this input and reply in JSON matching the response schema.\n\n"
                        + json.dumps(payload, default=str, sort_keys=True)
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=request_body,
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"Fireworks HTTP {response.status_code}"
                    if attempt < self.max_retries:
                        time.sleep(2**attempt)
                        continue
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise FireworksError("Fireworks response was truncated")
                content = choice.get("message", {}).get("content")
                if not content:
                    raise FireworksError("Fireworks returned empty structured content")
                return response_model.model_validate_json(_strip_code_fence(content))
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = f"Fireworks network error: {exc}"
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
            except FireworksError as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = exc.response.text[:300]
                raise FireworksError(f"Fireworks HTTP {status}: {detail}") from exc
            except ValidationError as exc:
                if attempt < self.max_retries:
                    request_body["messages"].extend(
                        [
                            {"role": "assistant", "content": content or "{}"},
                            {
                                "role": "user",
                                "content": (
                                    "Your JSON failed contract validation. Correct only the "
                                    "structured response and return valid JSON. Validation: "
                                    + str(exc)[:800]
                                ),
                            },
                        ]
                    )
                    continue
                raise FireworksError(
                    f"Invalid Fireworks structured response: {exc}"
                ) from exc
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise FireworksError(f"Invalid Fireworks structured response: {exc}") from exc

        raise FireworksError(last_error)


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text
