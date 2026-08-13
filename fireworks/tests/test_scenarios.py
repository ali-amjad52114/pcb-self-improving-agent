"""Mocked acceptance scenarios from the Fireworks scientist/critic brief."""

from fireworks.critic import critique_experiment
from fireworks.scientist import diagnose_and_propose
from fireworks.tests.mock_client import MockFireworksClient


def test_scenario_a_class_imbalance():
    proposal = diagnose_and_propose(
        current_config={"sampler": "standard"},
        metrics={"macro_f1": 0.72},
        per_class_metrics={
            "open_circuit": {"recall": 0.31},
            "short": {"recall": 0.84},
            "missing_hole": {"recall": 0.88},
        },
        dataset_summary={"open_circuit": "fewer training examples"},
        client=MockFireworksClient(),
    )
    assert proposal.action in {"change_sampler", "change_class_weights"}


def test_scenario_b_overfitting_prefers_augmentation():
    proposal = diagnose_and_propose(
        current_config={"augmentation": "basic"},
        metrics={"training_accuracy": 0.96, "validation_accuracy": 0.68},
        dataset_summary={
            "training_loss": "decreasing",
            "validation_loss": "increasing",
        },
        client=MockFireworksClient(),
    )
    assert proposal.action == "change_augmentation"


def test_scenario_c_avoids_previous_failed_model_scaling():
    proposal = diagnose_and_propose(
        current_config={"model": "resnet18"},
        metrics={"macro_f1": 0.62},
        recent_experiments=[
            {"action": "change_model", "verdict": "failed", "result": "worse"}
        ],
        client=MockFireworksClient(),
    )
    assert proposal.action != "change_model"


def test_scenario_d_uses_relevant_memory():
    proposal = diagnose_and_propose(
        current_config={"sampler": "standard"},
        metrics={"macro_f1": 0.61},
        per_class_metrics={"open_circuit": {"recall": 0.31}},
        dataset_summary={"class_balance": "minority open circuit"},
        retrieved_memories=[
            {
                "memory_id": "weighted-sampling-lesson",
                "failure": "Similar minority-class failure",
                "intervention": "weighted sampling",
                "result": "macro F1 improved from 0.61 to 0.74",
            }
        ],
        client=MockFireworksClient(),
    )
    assert proposal.action == "change_sampler"
    assert proposal.memory_used == ["weighted-sampling-lesson"]


def test_critic_computes_delta_and_extracts_lesson():
    critique = critique_experiment(
        before={"metrics": {"macro_f1": 0.62, "accuracy": 0.70}},
        experiment={
            "action": "change_sampler",
            "parameters": {"sampler": "weighted"},
            "hypothesis": "Class imbalance limits learning.",
        },
        after={"metrics": {"macro_f1": 0.73, "accuracy": 0.75}},
        client=MockFireworksClient(),
    )
    assert critique.verdict == "successful"
    assert critique.metric_delta["macro_f1"] == 0.11
    assert critique.hypothesis_supported is True
    assert critique.lesson
