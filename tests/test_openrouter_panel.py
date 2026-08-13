"""Tests for OpenRouter panel consensus and call budget (0 live API calls)."""

from __future__ import annotations

from pcb_agent.fakes.judge import FakeJudgeAdapter
from pcb_agent.integrations.judge_consensus import merge_panel_opinions


def test_consensus_majority_proceed():
    merged = merge_panel_opinions(
        [
            {
                "verdict": "agree",
                "confidence": 0.9,
                "recommendation": "proceed",
                "reasoning_summary": "ok",
                "_model": "a",
            },
            {
                "verdict": "agree",
                "confidence": 0.8,
                "recommendation": "proceed",
                "reasoning_summary": "ok",
                "_model": "b",
            },
            {
                "verdict": "disagree",
                "confidence": 0.4,
                "recommendation": "reconsider",
                "reasoning_summary": "no",
                "_model": "c",
            },
        ]
    )
    assert merged["recommendation"] == "proceed"
    assert merged["verdict"] == "agree"
    assert len(merged["panel"]) == 3
    assert merged["agreement_rate"] == 0.6667


def test_consensus_two_reconsider_wins():
    merged = merge_panel_opinions(
        [
            {
                "verdict": "disagree",
                "confidence": 0.7,
                "recommendation": "reconsider",
                "suggested_action_family": "change_sampler",
                "reasoning_summary": "try sampler",
                "_model": "a",
            },
            {
                "verdict": "disagree",
                "confidence": 0.8,
                "recommendation": "reconsider",
                "suggested_action_family": "change_sampler",
                "reasoning_summary": "try sampler",
                "_model": "b",
            },
            {
                "verdict": "agree",
                "confidence": 0.6,
                "recommendation": "proceed",
                "reasoning_summary": "fine",
                "_model": "c",
            },
        ]
    )
    assert merged["recommendation"] == "reconsider"
    assert merged["suggested_action_family"] == "change_sampler"
    assert merged["suggested_action_allowed"] is True


def test_consensus_rejects_invalid_suggested_action():
    merged = merge_panel_opinions(
        [
            {
                "verdict": "disagree",
                "confidence": 0.9,
                "recommendation": "reconsider",
                "suggested_action_family": "delete_production_db",
                "reasoning_summary": "bad",
                "_model": "a",
            }
        ]
    )
    assert merged["suggested_action_family"] == ""
    assert merged["suggested_action_allowed"] is False


def test_fake_judge_budget_exhausted():
    judge = FakeJudgeAdapter()
    opinion = judge.second_opinion(
        {
            "stage": "proposal_review",
            "trigger": "two_consecutive_failures",
            "_openrouter_calls": 5,
            "_openrouter_budget": 5,
        }
    )
    assert "openrouter_budget_exhausted" in opinion["risk_flags"]
    assert opinion.get("_budget_exhausted") is True
    assert opinion.get("_calls_made") == 0
    assert opinion["recommendation"] == "proceed"


def test_fake_judge_rich_schema_fields():
    judge = FakeJudgeAdapter(
        verdict="disagree",
        recommendation="reconsider",
        suggested_action_family="change_sampler",
        lesson_quality="strong",
        panel_size=2,
    )
    opinion = judge.second_opinion({"stage": "proposal_review", "trigger": "change_model"})
    assert opinion["suggested_action_family"] == "change_sampler"
    assert opinion["lesson_quality"] == "strong"
    assert len(opinion["panel"]) == 2
    assert opinion["_calls_made"] == 1
