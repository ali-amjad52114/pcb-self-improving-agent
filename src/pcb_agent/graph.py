"""Compile the PCB agent LangGraph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from pcb_agent.config import Settings
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.nodes.baseline import make_evaluate_baseline, make_train_baseline
from pcb_agent.nodes.critic import make_critique_result
from pcb_agent.nodes.experiment import make_evaluate_result, make_run_experiment
from pcb_agent.nodes.memory import make_retrieve_memory
from pcb_agent.nodes.persistence import make_store_experience
from pcb_agent.nodes.reasoning import (
    make_diagnose,
    make_judge_lesson,
    make_judge_proposal,
    make_propose_experiment,
)
from pcb_agent.routing import (
    route_after_critique,
    route_after_judge_proposal,
    route_after_propose,
    should_continue,
)
from pcb_agent.state import AgentState


def build_graph(deps: AgentDependencies, checkpointer: Any, settings: Settings):
    """Wire nodes and conditional edges. Dependencies are closed over, not in state."""

    openrouter_enabled = (
        bool(settings.openrouter_enabled)
        and bool(settings.openrouter_api_key)
        and bool(settings.openrouter_model)
        and settings.agent_mode != "fake"
    )

    graph = StateGraph(AgentState)

    graph.add_node("train_baseline", make_train_baseline(deps))
    graph.add_node("evaluate_baseline", make_evaluate_baseline(deps))
    graph.add_node("retrieve_memory", make_retrieve_memory(deps, settings))
    graph.add_node("diagnose", make_diagnose(deps))
    graph.add_node("propose_experiment", make_propose_experiment(deps))
    graph.add_node(
        "judge_proposal",
        make_judge_proposal(deps, openrouter_enabled=openrouter_enabled),
    )
    graph.add_node("run_experiment", make_run_experiment(deps))
    graph.add_node("evaluate_result", make_evaluate_result(deps))
    graph.add_node("critique_result", make_critique_result(deps))
    graph.add_node(
        "judge_lesson",
        make_judge_lesson(deps, openrouter_enabled=openrouter_enabled),
    )
    graph.add_node("store_experience", make_store_experience(deps))

    graph.add_edge(START, "train_baseline")
    graph.add_edge("train_baseline", "evaluate_baseline")
    graph.add_edge("evaluate_baseline", "retrieve_memory")
    graph.add_edge("retrieve_memory", "diagnose")
    graph.add_edge("diagnose", "propose_experiment")

    graph.add_conditional_edges(
        "propose_experiment",
        route_after_propose,
        {
            "judge_proposal": "judge_proposal",
            "run_experiment": "run_experiment",
        },
    )
    graph.add_conditional_edges(
        "judge_proposal",
        route_after_judge_proposal,
        {
            "propose_experiment": "propose_experiment",
            "run_experiment": "run_experiment",
        },
    )
    graph.add_edge("run_experiment", "evaluate_result")
    graph.add_edge("evaluate_result", "critique_result")

    def _after_critique(state: AgentState) -> str:
        return route_after_critique(state, openrouter_enabled=openrouter_enabled)

    graph.add_conditional_edges(
        "critique_result",
        _after_critique,
        {
            "judge_lesson": "judge_lesson",
            "store_experience": "store_experience",
        },
    )
    graph.add_edge("judge_lesson", "store_experience")

    graph.add_conditional_edges(
        "store_experience",
        should_continue,
        {
            "retrieve_memory": "retrieve_memory",
            "end": END,
        },
    )

    return graph.compile(checkpointer=checkpointer)
