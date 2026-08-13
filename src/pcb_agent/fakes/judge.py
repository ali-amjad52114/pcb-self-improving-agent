"""Deterministic OpenRouter stand-in — zero live API calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pcb_agent.contracts import JUDGE_FALLBACK


class FakeJudgeAdapter:
    """Configurable independent evaluator for tests and fake mode."""

    def __init__(
        self,
        *,
        verdict: str = "agree",
        recommendation: str = "proceed",
        confidence: float = 0.9,
        force_fallback: bool = False,
    ) -> None:
        self.verdict = verdict
        self.recommendation = recommendation
        self.confidence = confidence
        self.force_fallback = force_fallback
        self.call_count = 0
        self.last_context: dict[str, Any] | None = None

    def configure(
        self,
        *,
        verdict: str | None = None,
        recommendation: str | None = None,
        confidence: float | None = None,
        force_fallback: bool | None = None,
    ) -> None:
        if verdict is not None:
            self.verdict = verdict
        if recommendation is not None:
            self.recommendation = recommendation
        if confidence is not None:
            self.confidence = confidence
        if force_fallback is not None:
            self.force_fallback = force_fallback

    def second_opinion(self, context: dict) -> dict:
        self.call_count += 1
        self.last_context = deepcopy(context)
        if self.force_fallback:
            return dict(JUDGE_FALLBACK)
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "alternative_explanation": "",
            "risk_flags": [],
            "recommendation": self.recommendation,
            "reasoning_summary": "Proposal is supported by the supplied metrics.",
        }
