"""OpenRouter independent evaluator panel (Person 3)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from pcb_agent.config import Settings
from pcb_agent.contracts import (
    ALLOWED_ACTIONS,
    JUDGE_FALLBACK,
    normalize_judge,
)
from pcb_agent.fakes.judge import FakeJudgeAdapter
from pcb_agent.integrations.judge_consensus import merge_panel_opinions

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
If suggesting an alternative action family, it MUST be one of the allowed actions or empty.
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
        "evidence_citations": {"type": "array", "items": {"type": "string"}},
        "suggested_action_family": {
            "type": "string",
            "enum": sorted(ALLOWED_ACTIONS) + [""],
        },
        "suggested_action_allowed": {"type": "boolean"},
        "lesson_quality": {
            "type": "string",
            "enum": ["weak", "adequate", "strong"],
        },
    },
    "required": [
        "verdict",
        "confidence",
        "alternative_explanation",
        "risk_flags",
        "recommendation",
        "reasoning_summary",
        "evidence_citations",
        "suggested_action_family",
        "suggested_action_allowed",
        "lesson_quality",
    ],
    "additionalProperties": False,
}


def _budget_exhausted_result(*, spent: int, budget: int) -> dict[str, Any]:
    result = dict(JUDGE_FALLBACK)
    result["risk_flags"] = ["openrouter_budget_exhausted"]
    result["reasoning_summary"] = (
        f"Independent evaluator call budget exhausted ({spent}/{budget}); "
        "proceeding with primary scientist proposal."
    )
    result["_budget_exhausted"] = True
    return result


class OpenRouterJudge:
    """httpx-backed OpenRouter judge panel with budget + format ladder."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.call_count = 0
        self.ledger: list[dict[str, Any]] = []

    @property
    def chat_url(self) -> str:
        base = (self.settings.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip(
            "/"
        )
        return f"{base}/chat/completions"

    def remaining_budget(self, state_calls: int | None = None) -> int:
        spent = int(state_calls if state_calls is not None else self.call_count)
        return max(0, int(self.settings.openrouter_call_budget) - spent)

    def second_opinion(self, context: dict) -> dict:
        if not self.settings.openrouter_enabled:
            result = dict(JUDGE_FALLBACK)
            result["risk_flags"] = ["openrouter_disabled"]
            return normalize_judge(result)

        models = self.settings.openrouter_models()
        if not self.settings.openrouter_api_key or not models:
            result = dict(JUDGE_FALLBACK)
            result["risk_flags"] = ["openrouter_misconfigured"]
            result["reasoning_summary"] = (
                "OpenRouter disabled/unavailable (missing API key or model)."
            )
            return normalize_judge(result)

        # Budget is tracked via context._openrouter_calls when provided by nodes.
        spent = int(context.get("_openrouter_calls") or self.call_count)
        budget = int(self.settings.openrouter_call_budget)
        if spent >= budget:
            out = normalize_judge(_budget_exhausted_result(spent=spent, budget=budget))
            out["_budget_exhausted"] = True
            out["_calls_made"] = 0
            return out

        remaining = budget - spent
        models_to_call = models[: max(1, min(len(models), remaining))]

        opinions: list[dict[str, Any]] = []
        for model in models_to_call:
            absolute_spent = spent + len(opinions)
            if absolute_spent >= budget:
                break
            opinion = self._call_model_with_ladder(model=model, context=context)
            opinion["_model"] = model
            opinions.append(opinion)

        if not opinions:
            out = normalize_judge(_budget_exhausted_result(spent=spent, budget=budget))
            out["_budget_exhausted"] = True
            out["_calls_made"] = 0
            return out

        merged = merge_panel_opinions(opinions)
        merged["_panel_size"] = len(opinions)
        merged["_models"] = [o.get("_model") for o in opinions]
        merged["_calls_made"] = len(opinions)
        return merged

    def _call_model_with_ladder(self, *, model: str, context: dict) -> dict[str, Any]:
        """Try json_schema → json_object; record ledger entries."""
        for mode in ("json_schema", "json_object"):
            result = self._request_once(model=model, context=context, mode=mode)
            if result is not None:
                result["_response_format_mode"] = mode
                return result
        fallback = dict(JUDGE_FALLBACK)
        fallback["risk_flags"] = ["openrouter_format_ladder_exhausted", f"model={model}"]
        fallback["_model"] = model
        fallback["_response_format_mode"] = "fallback"
        return normalize_judge(fallback)

    def _request_once(
        self,
        *,
        model: str,
        context: dict,
        mode: str,
    ) -> dict[str, Any] | None:
        public_context = {
            k: v for k, v in context.items() if not str(k).startswith("_")
        }
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(self.settings.openrouter_max_tokens),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Evaluate this PCB experiment context and return structured JSON.\n"
                        f"Allowed action families: {sorted(ALLOWED_ACTIONS)}\n\n"
                        + json.dumps(public_context, default=str)
                    ),
                },
            ],
        }
        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_opinion",
                    "strict": True,
                    "schema": JUDGE_JSON_SCHEMA,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        provider: dict[str, Any] = {"require_parameters": True}
        order = self.settings.openrouter_provider_list()
        if order:
            provider["order"] = order
        payload["provider"] = provider

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pcb-self-improving-agent",
            "X-Title": "PCB Self-Improving Agent",
        }

        max_retries = max(0, int(self.settings.openrouter_max_retries))
        attempt = 0
        last_error = "unknown"
        stage = str(context.get("stage") or "unknown")
        trigger = str(context.get("trigger") or "")

        while attempt <= max_retries:
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.settings.openrouter_timeout_seconds) as client:
                    response = client.post(self.chat_url, headers=headers, json=payload)
                latency_ms = int((time.perf_counter() - started) * 1000)
                self.call_count += 1

                tokens = None
                try:
                    usage = response.json().get("usage") or {}
                    tokens = usage.get("total_tokens")
                except Exception:
                    tokens = None

                ledger_entry = {
                    "model": model,
                    "stage": stage,
                    "trigger": trigger,
                    "latency_ms": latency_ms,
                    "tokens": tokens,
                    "http_status": response.status_code,
                    "response_format_mode": mode,
                    "fallback": False,
                }

                if response.status_code in {401, 402}:
                    ledger_entry["fallback"] = True
                    self.ledger.append(ledger_entry)
                    result = dict(JUDGE_FALLBACK)
                    result["risk_flags"] = [f"openrouter_http_{response.status_code}"]
                    return normalize_judge(result)

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"http_{response.status_code}"
                    self.ledger.append(ledger_entry)
                    if attempt >= max_retries:
                        return None
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(min(30, int(retry_after)))
                    else:
                        time.sleep(2**attempt)
                    attempt += 1
                    continue

                if response.status_code >= 400:
                    # Likely unsupported structured output for this mode.
                    ledger_entry["fallback"] = True
                    self.ledger.append(ledger_entry)
                    return None

                body = response.json()
                content = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    return None
                data = normalize_judge(parsed)
                self.ledger.append(ledger_entry)
                data["_model"] = model
                data["_response_format_mode"] = mode
                return data

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = str(exc)
                self.ledger.append(
                    {
                        "model": model,
                        "stage": stage,
                        "trigger": trigger,
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "tokens": None,
                        "http_status": None,
                        "response_format_mode": mode,
                        "fallback": True,
                        "error": last_error[:120],
                    }
                )
                if attempt >= max_retries:
                    return None
                time.sleep(2**attempt)
                attempt += 1
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
                self.ledger.append(
                    {
                        "model": model,
                        "stage": stage,
                        "trigger": trigger,
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "tokens": None,
                        "http_status": None,
                        "response_format_mode": mode,
                        "fallback": True,
                        "error": f"invalid_json:{exc}"[:120],
                    }
                )
                return None
            except Exception as exc:
                last_error = str(exc)
                self.ledger.append(
                    {
                        "model": model,
                        "stage": stage,
                        "trigger": trigger,
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "tokens": None,
                        "http_status": None,
                        "response_format_mode": mode,
                        "fallback": True,
                        "error": last_error[:120],
                    }
                )
                return None

        return None


def build_judge(settings: Settings):
    """Build real OpenRouter judge when enabled+configured; else fake."""
    if (
        settings.openrouter_enabled
        and settings.openrouter_api_key
        and settings.openrouter_models()
    ):
        return OpenRouterJudge(settings)
    return FakeJudgeAdapter()
