"""Standalone, no-cost demo: python -m fireworks.tests.demo."""

from fireworks.critic import critique_experiment
from fireworks.scientist import diagnose_and_propose
from fireworks.tests.mock_client import MockFireworksClient


def main() -> None:
    client = MockFireworksClient()
    before = {
        "config": {
            "model": "resnet18",
            "learning_rate": 0.001,
            "batch_size": 32,
            "sampler": "standard",
        },
        "metrics": {"macro_f1": 0.62, "accuracy": 0.70},
        "per_class_metrics": {
            "open_circuit": {"recall": 0.31},
            "short": {"recall": 0.84},
            "missing_hole": {"recall": 0.88},
        },
    }
    proposal = diagnose_and_propose(
        current_config=before["config"],
        metrics=before["metrics"],
        per_class_metrics=before["per_class_metrics"],
        dataset_summary={"open_circuit": "significantly fewer training examples"},
        client=client,
    )

    print(f"CURRENT F1: {before['metrics']['macro_f1']:.2f}\n")
    print(f"DIAGNOSIS:\n{proposal.diagnosis}\n")
    print(f"HYPOTHESIS:\n{proposal.hypothesis}\n")
    print(
        "PROPOSED EXPERIMENT:\n"
        f"{proposal.action} -> {proposal.parameters}\n"
    )
    print(f"CONFIDENCE:\n{proposal.confidence:.2f}\n")

    after = {
        "config": {**before["config"], "sampler": "weighted"},
        "metrics": {"macro_f1": 0.73, "accuracy": 0.75},
        "per_class_metrics": {"open_circuit": {"recall": 0.66}},
    }
    critique = critique_experiment(
        before=before,
        experiment=proposal.model_dump(),
        after=after,
        client=client,
    )
    print(f"NEW F1: {after['metrics']['macro_f1']:.2f}\n")
    print(f"VERDICT:\n{critique.verdict.replace('_', ' ').title()}\n")
    print(f"LESSON:\n{critique.lesson}")


if __name__ == "__main__":
    main()
