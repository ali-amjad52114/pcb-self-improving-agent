"""Tests for judge reconsider routing."""

from pcb_agent.routing import route_after_judge_proposal, should_reconsider


def test_proceed_routes_to_run_experiment():
    state = {
        "evaluator_opinion": {"recommendation": "proceed"},
        "proposal_revision_count": 0,
    }
    assert should_reconsider(state) is False
    assert route_after_judge_proposal(state) == "run_experiment"


def test_reconsider_with_zero_revisions_routes_to_propose():
    state = {
        "evaluator_opinion": {"recommendation": "reconsider"},
        "proposal_revision_count": 0,
    }
    assert should_reconsider(state) is True
    assert route_after_judge_proposal(state) == "propose_experiment"


def test_reconsider_with_one_revision_routes_to_run():
    state = {
        "evaluator_opinion": {"recommendation": "reconsider"},
        "proposal_revision_count": 1,
    }
    assert should_reconsider(state) is False
    assert route_after_judge_proposal(state) == "run_experiment"
