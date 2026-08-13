"""Tests for stop conditions."""

from pcb_agent.routing import should_continue, stop_reason_for, will_stop_after_iteration


def test_continue_when_below_target_with_budget():
    state = {
        "current_metrics": {"macro_f1": 0.83},
        "target_metric": "macro_f1",
        "target_value": 0.84,
        "iteration": 1,
        "experiment_budget": 4,
    }
    assert stop_reason_for(state) is None
    assert should_continue(state) == "retrieve_memory"
    assert will_stop_after_iteration(state) is False


def test_stop_target_reached():
    state = {
        "current_metrics": {"macro_f1": 0.84},
        "target_metric": "macro_f1",
        "target_value": 0.84,
        "iteration": 2,
        "experiment_budget": 4,
    }
    assert stop_reason_for(state) == "target_reached"
    assert should_continue(state) == "end"


def test_stop_budget_exhausted():
    state = {
        "current_metrics": {"macro_f1": 0.70},
        "target_metric": "macro_f1",
        "target_value": 0.84,
        "iteration": 4,
        "experiment_budget": 4,
    }
    assert stop_reason_for(state) == "experiment_budget_exhausted"
    assert should_continue(state) == "end"


def test_will_stop_after_iteration_on_budget_edge():
    # iteration=3 means after store → 4, which exhausts budget 4
    state = {
        "current_metrics": {"macro_f1": 0.70},
        "target_metric": "macro_f1",
        "target_value": 0.84,
        "iteration": 3,
        "experiment_budget": 4,
    }
    assert will_stop_after_iteration(state) is True
