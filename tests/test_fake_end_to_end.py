"""End-to-end fake graph — zero external API calls."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from pcb_agent.config import Settings
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.fakes.judge import FakeJudgeAdapter
from pcb_agent.fakes.memory import FakeMemory
from pcb_agent.fakes.ml import FakeML
from pcb_agent.fakes.reasoning import FakeReasoning
from pcb_agent.graph import build_graph


def _settings(**overrides) -> Settings:
    base = {
        "agent_mode": "fake",
        "target_metric": "macro_f1",
        "target_value": 0.84,
        "experiment_budget": 4,
        "checkpointer_backend": "memory",
        "openrouter_enabled": False,
        "openrouter_call_budget": 5,
        "memory_top_k": 5,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _initial(run_id: str, settings: Settings) -> dict:
    config = {
        "model_family": "resnet18",
        "learning_rate": 0.001,
        "batch_size": 32,
        "image_size": 224,
        "augmentation": "basic",
        "sampler": "random",
        "class_weights": None,
        "confidence_threshold": 0.5,
    }
    return {
        "run_id": run_id,
        "iteration": 0,
        "experiment_budget": settings.experiment_budget,
        "target_metric": settings.target_metric,
        "target_value": settings.target_value,
        "dataset_summary": {
            "class_imbalance_ratio": 5.4,
            "minority_class": "open_circuit",
        },
        "current_config": config,
        "best_config": config,
        "current_model_id": "",
        "baseline_metrics": {},
        "previous_metrics": {},
        "current_metrics": {},
        "best_metrics": {},
        "confusion_matrix": {},
        "per_class_metrics": {},
        "memory_query": "",
        "retrieved_lessons": [],
        "diagnosis": {},
        "proposed_experiment": {},
        "proposal_revision_count": 0,
        "evaluator_opinion": {},
        "evaluator_trigger": "",
        "critique": {},
        "experiment_history": [],
        "consecutive_failures": 0,
        "last_metric_delta": 0.0,
        "last_experiment_helped": False,
        "status": "started",
        "stop_reason": "",
        "warnings": [],
        "pending_experiment_id": "",
        "openrouter_calls": 0,
        "openrouter_call_budget": settings.openrouter_call_budget,
        "openrouter_ledger": [],
        "judge_skipped": False,
        "judge_skip_reason": "",
    }


def test_fake_end_to_end_terminates_and_improves():
    settings = _settings()
    memory = FakeMemory()
    deps = AgentDependencies(
        memory=memory,
        ml=FakeML(),
        reasoning=FakeReasoning(),
        judge=FakeJudgeAdapter(),
    )
    graph = build_graph(deps, InMemorySaver(), settings)
    final = graph.invoke(
        _initial("run_cold_test", settings),
        config={"configurable": {"thread_id": "run_cold_test"}},
    )

    assert final["status"] == "complete"
    assert final["stop_reason"] in {"target_reached", "experiment_budget_exhausted"}
    history = final["experiment_history"]
    assert len(history) > 0
    assert len(history) <= settings.experiment_budget
    assert len(memory.lessons) == len(history)
    assert final["best_metrics"]["macro_f1"] >= final["baseline_metrics"]["macro_f1"]
    assert deps.judge.call_count == 0  # type: ignore[attr-defined]


def test_memory_changes_proposal_sequence():
    settings = _settings()
    shared_memory = FakeMemory()

    cold_deps = AgentDependencies(
        memory=shared_memory,
        ml=FakeML(),
        reasoning=FakeReasoning(),
        judge=FakeJudgeAdapter(),
    )
    cold_graph = build_graph(cold_deps, InMemorySaver(), settings)
    cold = cold_graph.invoke(
        _initial("run_cold_mem", settings),
        config={"configurable": {"thread_id": "run_cold_mem"}},
    )
    cold_actions = [h["proposal"]["next_action"] for h in cold["experiment_history"]]

    # Seed an explicit failure lesson for batch size so warm run avoids it.
    shared_memory.store_lesson(
        {
            "lesson_id": "lesson_bad_batch",
            "failure_summary": "Poor recall on minority open-circuit class",
            "intervention": "larger batch size",
            "result": "-0.01 macro F1",
            "confidence": 0.87,
            "helped": False,
            "next_action": "change_batch_size",
        }
    )
    shared_memory.store_lesson(
        {
            "lesson_id": "lesson_good_sampler",
            "failure_summary": "Poor recall on minority open-circuit class",
            "intervention": "weighted sampler",
            "result": "+0.09 macro F1",
            "confidence": 0.91,
            "helped": True,
            "next_action": "change_sampler",
        }
    )

    warm_deps = AgentDependencies(
        memory=shared_memory,
        ml=FakeML(),
        reasoning=FakeReasoning(),
        judge=FakeJudgeAdapter(),
    )
    warm_graph = build_graph(warm_deps, InMemorySaver(), settings)
    warm = warm_graph.invoke(
        _initial("run_warm_mem", settings),
        config={"configurable": {"thread_id": "run_warm_mem"}},
    )
    warm_actions = [h["proposal"]["next_action"] for h in warm["experiment_history"]]

    assert cold_actions[0] == "change_learning_rate"
    assert warm_actions[0] == "change_sampler"
    assert warm_actions != cold_actions
    assert len(warm_actions) <= len(cold_actions)
    assert warm["best_metrics"]["macro_f1"] >= warm["baseline_metrics"]["macro_f1"]
