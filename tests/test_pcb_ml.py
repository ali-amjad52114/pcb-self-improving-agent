from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from pcb_agent.contracts import ProposalValidationError
from pcb_agent.integrations import pcb_ml


def _dataset(tmp_path: Path) -> Path:
    splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    colors = {"open": (235, 25, 25), "short": (25, 25, 235)}
    for split, count in (("train", 8), ("validation", 3), ("test", 2)):
        for label, color in colors.items():
            for index in range(count):
                path = tmp_path / f"{split}_{label}_{index}.png"
                Image.new("RGB", (24, 24), color).save(path)
                splits[split].append({"image": path.name, "label": label})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"classes": ["open", "short"], "splits": splits}),
        encoding="utf-8",
    )
    return manifest


def test_train_and_evaluate_real_images(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path)
    result = pcb_ml.train_model(
        {
            "dataset_manifest": str(manifest),
            "model_dir": str(tmp_path / "models"),
            "epochs": 3,
            "learning_rate": 0.01,
            "batch_size": 4,
            "seed": 7,
        }
    )

    evaluated = pcb_ml.evaluate_model(result["model_id"])
    assert evaluated["metrics"] == {"macro_f1": 1.0, "accuracy": 1.0}
    assert evaluated["per_class_metrics"]["open"]["support"] == 3
    assert evaluated["confusion_matrix"]["open"]["open"] == 3
    assert evaluated["training_history"]["training_loss"]
    assert evaluated["training_history"]["validation_loss"] >= 0
    assert evaluated["misclassified_examples"] == []

    test_result = pcb_ml.evaluate_test_model(result["model_id"])
    assert test_result["evaluation_split"] == "test"
    assert test_result["per_class_metrics"]["short"]["support"] == 2

    summary = pcb_ml.get_dataset_summary({"dataset_manifest": str(manifest)})
    assert summary["total_samples"] == 26
    assert summary["train_samples"] == 16
    assert summary["validation_samples"] == 6
    assert summary["test_samples"] == 4
    assert summary["class_distribution"] == {"open": 13, "short": 13}
    assert summary["train_class_distribution"] == {"open": 8, "short": 8}


def test_evaluate_test_trains_best_config(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path)
    result = pcb_ml.evaluate_test(
        {
            "dataset_manifest": str(manifest),
            "model_dir": str(tmp_path / "models"),
            "epochs": 2,
            "learning_rate": 0.01,
        }
    )
    assert result["evaluation_split"] == "test"
    assert result["metrics"]["macro_f1"] == 1.0
    assert result["model_id"].startswith("pcb_")


@pytest.mark.parametrize(
    ("action", "parameters", "key", "expected"),
    [
        ("change_sampler", {}, "sampler", "weighted"),
        ("change_class_weights", {}, "class_weights", "balanced"),
        ("increase_epochs", {"epochs": 9}, "epochs", 9),
        ("increase_epochs", {"additional_epochs": 4}, "epochs", 7),
        ("change_learning_rate", {"learning_rate": 0.02}, "learning_rate", 0.02),
        ("change_model", {"model": "resnet34"}, "model_family", "resnet34"),
        (
            "change_confidence_threshold",
            {"confidence_threshold": 0.3},
            "confidence_threshold",
            0.3,
        ),
    ],
)
def test_apply_supported_experiment(action, parameters, key, expected) -> None:
    original = {"epochs": 3, "learning_rate": 0.001}
    updated = pcb_ml.apply_experiment(
        original, {"next_action": action, "parameters": parameters}
    )
    assert updated[key] == expected
    assert updated["_last_action"] == action
    assert "_last_action" not in original


def test_rejects_arbitrary_experiment() -> None:
    with pytest.raises(ProposalValidationError):
        pcb_ml.apply_experiment({}, {"next_action": "execute_python", "parameters": {}})


def test_manifest_requires_validation_data(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"train": [{"image": "x", "label": "a"}]}))
    with pytest.raises(ValueError, match="non-empty train and validation"):
        pcb_ml.train_model({"dataset_manifest": str(manifest)})
