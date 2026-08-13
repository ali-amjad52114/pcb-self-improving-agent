"""OpenRouter independent evaluator (Person 3)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from pcb_agent.config import Settings
from pcb_agent.contracts import JUDGE_FALLBACK, JudgeOpinion, normalize_judge
from pcb_agent.fakes.judge import FakeJudgeAdapter

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an independent ML experiment evaluator.

You are NOT the primary scientist and you do not execute tools.

Review the provided PCB classifier evidence.

Determine whether the scientist's diagnosis, proposed intervention, or final lesson is supported by the measured evidence.

Be skeptical but practical.

Do not invent metrics.
Do not claim an intervention worked unless the supplied before/after metrics support it.
Do not produce hidden chain-of-thought.
Return only the requested structured JSON.

"reconsider" should be used only when there is a meaningful reason not to proceed.
"""

JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["agree", "disagree", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "alternative_explanation": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "recommendation": {
            "type": "string",
            "enum": ["proceed", "reconsider"],
        },
        "reasoning_summary": {"type": "string"},
    },
    "required": [
        "verdict",
        "confidence",
        "alternative_explanation",
        "risk_flags",
        "recommendation",
        "reasoning_summary",
    ],
    "additionalProperties": False,
}


class OpenRouterJudge:
    """httpx-backed OpenRouter judge with safe fallbacks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.call_count = 0

    def second_opinion(self, context: dict) -> dict:
        if not self.settings.openrouter_enabled:
            result = dict(JUDGE_FALLBACK)
            result["risk_flags"] = ["openrouter_disabled"]
            return result
        if not self.settings.openrouter_api_key or not self.settings.openrouter_model:
            result = dict(JUDGE_FALLBACK)
            result["risk_flags"] = ["openrouter_misconfigured"]
            result["reasoning_summary"] = (
                "OpenRouter disabled/unavailable (missing API key or model)."
            )
            return result

        payload = {
            "model": self.settings.openrouter_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Evaluate this PCB experiment context and return structured JSON.\n\n"
                        + json.dumps(context, default=str)
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_opinion",
                    "strict": True,
                    "schema": JUDGE_JSON_SCHEMA,
                },
            },
            "provider": {
                "require_parameters": True,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pcb-self-improving-agent",
            "X-Title": "PCB Self-Improving Agent",
        }

        max_retries = max(0, int(self.settings.openrouter_max_retries))
        attempt = 0
        last_error = "unknown"

        while attempt <= max_retries:
            try:
                with httpx.Client(timeout=self.settings.openrouter_timeout_seconds) as client:
                    response = client.post(OPENROUTER_URL, headers=headers, json=payload)
                self.call_count += 1

                if response.status_code in {401, 402}:
                    result = dict(JUDGE_FALLBACK)
                    result["risk_flags"] = [f"openrouter_http_{response.status_code}"]
                    return result

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"http_{response.status_code}"
                    if attempt >= max_retries:
                        break
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(min(30, int(retry_after)))
                    else:
                        time.sleep(2**attempt)
                    attempt += 1
                    continue

                if response.status_code >= 400:
                    result = dict(JUDGE_FALLBACK)
                    result["risk_flags"] = [f"openrouter_http_{response.status_code}"]
                    return result

                body = response.json()
                content = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if isinstance(content, list):
                    # Some providers return content parts.
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                parsed = json.loads(content)
                validated = JudgeOpinion.model_validate(parsed)
                return validated.model_dump()

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = str(exc)
                if attempt >= max_retries:
                    break
                time.sleep(2**attempt)
                attempt += 1
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
                result = dict(JUDGE_FALLBACK)
                result["risk_flags"] = ["openrouter_invalid_json", str(exc)[:80]]
                return normalize_judge(result)
            except Exception as exc:
                last_error = str(exc)
                break

        result = dict(JUDGE_FALLBACK)
        result["risk_flags"] = ["openrouter_unavailable", last_error[:80]]
        return result


def build_judge(settings: Settings):
    """Prefer OpenRouter when enabled; otherwise fake judge (still 0 live calls)."""
    if settings.openrouter_enabled and settings.openrouter_api_key and settings.openrouter_model:
        return OpenRouterJudge(settings)
    return FakeJudgeAdapter()
