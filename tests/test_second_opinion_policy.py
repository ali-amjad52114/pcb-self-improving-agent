"""Tests for OpenRouter / second-opinion trigger policy."""

from pcb_agent.routing import proposal_judge_reason


def test_high_confidence_skips_judge():
    state = {
        "diagnosis": {"confidence": 0.8},
        "proposed_experiment": {
            "confidence": 0.8,
            "next_action": "change_sampler",
        },
        "consecutive_failures": 0,
    }
    assert proposal_judge_reason(state) is None


def test_low_diagnosis_confidence_triggers_judge():
    state = {
        "diagnosis": {"confidence": 0.4},
        "proposed_experiment": {
            "confidence": 0.9,
            "next_action": "change_sampler",
        },
        "consecutive_failures": 0,
    }
    assert proposal_judge_reason(state) == "low_diagnosis_confidence"


def test_low_proposal_confidence_triggers_judge():
    state = {
        "diagnosis": {"confidence": 0.9},
        "proposed_experiment": {
            "confidence": 0.4,
            "next_action": "change_sampler",
        },
        "consecutive_failures": 0,
    }
    assert proposal_judge_reason(state) == "low_proposal_confidence"


def test_two_consecutive_failures_triggers_judge():
    state = {
        "diagnosis": {"confidence": 0.9},
        "proposed_experiment": {
            "confidence": 0.9,
            "next_action": "change_sampler",
        },
        "consecutive_failures": 2,
    }
    assert proposal_judge_reason(state) == "two_consecutive_failures"


def test_change_model_triggers_judge():
    state = {
        "diagnosis": {"confidence": 0.9},
        "proposed_experiment": {
            "confidence": 0.9,
            "next_action": "change_model",
        },
        "consecutive_failures": 0,
    }
    assert proposal_judge_reason(state) == "model_family_switch"
