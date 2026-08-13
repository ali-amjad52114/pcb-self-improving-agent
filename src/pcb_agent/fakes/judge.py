"""Deterministic OpenRouter stand-in — zero live API calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pcb_agent.contracts import JUDGE_FALLBACK, normalize_judge


class FakeJudgeAdapter:
    """Configurable independent evaluator for tests and fake mode."""

    def __init__(
        self,
        *,
        verdict: str = "agree",
        recommendation: str = "proceed",
        confidence: float = 0.9,
        force_fallback: bool = False,
        suggested_action_family: str = "",
        lesson_quality: str = "adequate",
        panel_size: int = 1,
    ) -> None:
        self.verdict = verdict
        self.recommendation = recommendation
        self.confidence = confidence
        self.force_fallback = force_fallback
        self.suggested_action_family = suggested_action_family
        self.lesson_quality = lesson_quality
        self.panel_size = panel_size
        self.call_count = 0
        self.ledger: list[dict[str, Any]] = []
        self.last_context: dict[str, Any] | None = None

    def configure(
        self,
        *,
        verdict: str | None = None,
        recommendation: str | None = None,
        confidence: float | None = None,
        force_fallback: bool | None = None,
        suggested_action_family: str | None = None,
        lesson_quality: str | None = None,
    ) -> None:
        if verdict is not None:
            self.verdict = verdict
        if recommendation is not None:
            self.recommendation = recommendation
        if confidence is not None:
            self.confidence = confidence
        if force_fallback is not None:
            self.force_fallback = force_fallback
        if suggested_action_family is not None:
            self.suggested_action_family = suggested_action_family
        if lesson_quality is not None:
            self.lesson_quality = lesson_quality

    def second_opinion(self, context: dict) -> dict:
        spent = int(context.get("_openrouter_calls") or self.call_count)
        budget = int(context.get("_openrouter_budget") or 10_000)
        if spent >= budget:
            result = dict(JUDGE_FALLBACK)
            result["risk_flags"] = ["openrouter_budget_exhausted"]
            result["reasoning_summary"] = (
                f"Independent evaluator call budget exhausted ({spent}/{budget}); "
                "proceeding with primary scientist proposal."
            )
            out = normalize_judge(result)
            out["_budget_exhausted"] = True
            out["_calls_made"] = 0
            return out

        self.call_count += 1
        self.last_context = deepcopy(context)
        self.ledger.append(
            {
                "model": "fake-judge",
                "stage": context.get("stage"),
                "trigger": context.get("trigger"),
                "latency_ms": 0,
                "tokens": 0,
                "response_format_mode": "fake",
                "fallback": self.force_fallback,
            }
        )
        if self.force_fallback:
            out = normalize_judge(dict(JUDGE_FALLBACK))
            out["_calls_made"] = 1
            return out

        panel = [
            {
                "model": f"fake-{i+1}",
                "verdict": self.verdict,
                "recommendation": self.recommendation,
                "confidence": self.confidence,
                "response_format_mode": "fake",
            }
            for i in range(max(1, self.panel_size))
        ]
        out = normalize_judge(
            {
                "verdict": self.verdict,
                "confidence": self.confidence,
                "alternative_explanation": "",
                "risk_flags": [],
                "recommendation": self.recommendation,
                "reasoning_summary": "Proposal is supported by the supplied metrics.",
                "evidence_citations": ["fake_metrics_present"],
                "suggested_action_family": self.suggested_action_family,
                "suggested_action_allowed": bool(self.suggested_action_family),
                "lesson_quality": self.lesson_quality,
                "panel": panel,
                "agreement_rate": 1.0,
                "dissent_summary": "",
            }
        )
        out["_calls_made"] = 1
        out["_panel_size"] = len(panel)
        return out
