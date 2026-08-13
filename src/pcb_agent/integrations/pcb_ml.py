"""Small, real NumPy image classifier implementing the frozen ``MLPort``.

The dataset preparation step writes a JSON split manifest.  This module keeps
the graph independent of the dataset layout and deliberately has no framework
dependency: Pillow extracts compact image features and a mini-batch softmax
classifier is trained with NumPy.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from pcb_agent.contracts import ALLOWED_ACTIONS, ProposalValidationError


DEFAULT_MANIFEST = Path("data/pcb_splits.json")
DEFAULT_MODEL_DIR = Path("data/models")
_MODEL_PATHS: dict[str, Path] = {}


def _manifest_path(config: dict[str, Any]) -> Path:
    value = config.get("dataset_manifest") or os.getenv("PCB_DATASET_MANIFEST")
    return Path(value or DEFAULT_MANIFEST).expanduser().resolve()


def _load_manifest(path: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"PCB split manifest not found at {path}. Run the dataset preparation step "
            "or set PCB_DATASET_MANIFEST."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    split_source = raw.get("splits", raw)
    splits: dict[str, list[dict[str, str]]] = {}
    labels: set[str] = set()
    for split in ("train", "validation", "test"):
        records: list[dict[str, str]] = []
        for item in split_source.get(split, []):
            image = item.get("image") or item.get("image_path") or item.get("path")
            label = item.get("label") or item.get("class") or item.get("class_name")
            if not image or label is None:
                raise ValueError(f"Invalid {split} manifest record: {item!r}")
            image_path = Path(str(image))
            if not image_path.is_absolute():
                # Dataset manifests may be project-relative (data/pcb_crops/...)
                # or manifest-relative (local fixture filenames).
                project_relative = image_path.resolve()
                image_path = (
                    project_relative
                    if project_relative.is_file()
                    else path.parent / image_path
                )
            record = {"image": str(image_path.resolve()), "label": str(label)}
            records.append(record)
            labels.add(record["label"])
        splits[split] = records
    if not splits["train"] or not splits["validation"]:
        raise ValueError("PCB manifest must contain non-empty train and validation splits")
    classes = [str(value) for value in raw.get("classes", [])]
    classes = classes or sorted(labels)
    unknown = labels.difference(classes)
    if unknown:
        raise ValueError(f"Manifest classes omit observed labels: {sorted(unknown)}")
    return splits, classes


def _feature_shape(config: dict[str, Any]) -> tuple[int, int]:
    family = str(config.get("model_family") or config.get("model") or "resnet18")
    side = 12 if family in {"efficientnet_b0", "resnet34", "resnet50"} else 8
    requested = max(8, int(config.get("image_size", 224)))
    if requested >= 256:
        side += 2
    elif requested <= 128:
        side = max(6, side - 2)
    return min(side, requested), 8


def _features(image: Image.Image, config: dict[str, Any]) -> np.ndarray:
    side, bins = _feature_shape(config)
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    small = np.asarray(rgb.resize((side, side)), dtype=np.float32) / 255.0
    gray = (
        0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
    ).reshape(-1)
    histograms = [
        np.histogram(small[:, :, channel], bins=bins, range=(0.0, 1.0))[0]
        for channel in range(3)
    ]
    histogram = np.concatenate(histograms).astype(np.float32)
    histogram /= max(1.0, float(side * side))
    return np.concatenate([gray, histogram, small.mean(axis=(0, 1)), small.std(axis=(0, 1))])


def _load_xy(
    records: list[dict[str, str]], classes: list[str], config: dict[str, Any], *, train: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_to_index = {label: index for index, label in enumerate(classes)}
    vectors: list[np.ndarray] = []
    targets: list[int] = []
    paths: list[str] = []
    augmentation = str(config.get("augmentation") or "none").lower()
    for record in records:
        image_path = record["image"]
        with Image.open(image_path) as image:
            vectors.append(_features(image, config))
            targets.append(class_to_index[record["label"]])
            paths.append(image_path)
            if train and augmentation in {"basic", "strong"}:
                vectors.append(_features(ImageOps.mirror(image), config))
                targets.append(class_to_index[record["label"]])
                paths.append(image_path)
            if train and augmentation == "strong":
                vectors.append(_features(ImageEnhance.Contrast(image).enhance(1.3), config))
                targets.append(class_to_index[record["label"]])
                paths.append(image_path)
    return np.stack(vectors), np.asarray(targets, dtype=np.int64), paths


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (x - mean) / scale, mean, scale


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _loss(probs: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    chosen = np.clip(probs[np.arange(len(y)), y], 1e-8, 1.0)
    return float(np.average(-np.log(chosen), weights=weights[y]))


def _class_weights(y: np.ndarray, class_count: int, setting: Any) -> np.ndarray:
    if setting in (None, False, "none"):
        return np.ones(class_count, dtype=np.float32)
    counts = np.bincount(y, minlength=class_count).astype(np.float32)
    balanced = len(y) / (class_count * np.maximum(counts, 1.0))
    if isinstance(setting, dict):
        # A dict is accepted only when keys are integer class indices; named
        # weights are resolved by apply_experiment into the balanced strategy.
        for key, value in setting.items():
            if str(key).isdigit() and int(key) < class_count:
                balanced[int(key)] = float(value)
    return balanced


def train_model(config: dict) -> dict:
    """Train a real image classifier on the manifest's training split."""
    config = deepcopy(config)
    manifest_path = _manifest_path(config)
    splits, classes = _load_manifest(manifest_path)
    x, y, _ = _load_xy(splits["train"], classes, config, train=True)
    x, mean, scale = _standardize_fit(x)
    rng = np.random.default_rng(int(config.get("seed", 42)))
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], len(classes))).astype(np.float32)
    bias = np.zeros(len(classes), dtype=np.float32)
    learning_rate = float(config.get("learning_rate", 0.001))
    # Compact handcrafted features converge at a larger effective step than a CNN.
    learning_rate = min(0.5, max(1e-4, learning_rate * 30.0))
    batch_size = max(1, int(config.get("batch_size", 32)))
    epochs = max(1, int(config.get("epochs", 3)))
    sample_weights = _class_weights(y, len(classes), config.get("class_weights"))
    sampler = str(config.get("sampler") or "random").lower()
    history: list[float] = []
    indices = np.arange(len(y))
    per_class = [indices[y == index] for index in range(len(classes))]
    for _epoch in range(epochs):
        if sampler in {"weighted", "balanced", "weighted_random"}:
            probabilities = 1.0 / np.maximum(np.bincount(y, minlength=len(classes))[y], 1)
            probabilities = probabilities / probabilities.sum()
            epoch_indices = rng.choice(indices, size=len(indices), replace=True, p=probabilities)
        else:
            epoch_indices = rng.permutation(indices)
        for start in range(0, len(epoch_indices), batch_size):
            batch = epoch_indices[start : start + batch_size]
            xb, yb = x[batch], y[batch]
            probs = _softmax(xb @ weights + bias)
            gradient = probs
            gradient[np.arange(len(batch)), yb] -= 1.0
            gradient *= sample_weights[yb, None]
            gradient /= max(1, len(batch))
            weights -= learning_rate * (xb.T @ gradient + 1e-4 * weights)
            bias -= learning_rate * gradient.sum(axis=0)
        history.append(_loss(_softmax(x @ weights + bias), y, sample_weights))

    model_dir = Path(config.get("model_dir") or os.getenv("PCB_MODEL_DIR") or DEFAULT_MODEL_DIR)
    model_dir = model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = str(config.get("_experiment_id") or "")
    fingerprint = hashlib.sha256(
        (experiment_id + json.dumps(config, sort_keys=True, default=str)).encode("utf-8")
    ).hexdigest()[:16]
    model_id = f"pcb_{experiment_id or fingerprint}"
    model_path = model_dir / f"{model_id}.npz"
    np.savez_compressed(
        model_path,
        weights=weights,
        bias=bias,
        mean=mean,
        scale=scale,
        classes=np.asarray(classes),
        config_json=np.asarray(json.dumps(config, default=str)),
        manifest_path=np.asarray(str(manifest_path)),
        training_loss=np.asarray(history),
        class_counts=np.bincount(y, minlength=len(classes)),
    )
    _MODEL_PATHS[model_id] = model_path
    return {
        "model_id": model_id,
        "metadata": {
            "source": "pcb_numpy_softmax",
            "training_samples": len(y),
            "classes": classes,
            "training_loss": history,
            "model_path": str(model_path),
        },
    }


def _find_model(model_id: str) -> Path:
    if model_id in _MODEL_PATHS and _MODEL_PATHS[model_id].is_file():
        return _MODEL_PATHS[model_id]
    model_dir = Path(os.getenv("PCB_MODEL_DIR") or DEFAULT_MODEL_DIR).expanduser().resolve()
    path = model_dir / f"{model_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_id}")
    return path


def _metrics(
    y: np.ndarray, predictions: np.ndarray, classes: list[str]
) -> tuple[dict[str, float], dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    count = len(classes)
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (y, predictions), 1)
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for index, label in enumerate(classes):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(matrix[index, :].sum()),
        }
        f1_values.append(f1)
    confusion = {
        true_label: {
            predicted_label: int(matrix[true_index, predicted_index])
            for predicted_index, predicted_label in enumerate(classes)
        }
        for true_index, true_label in enumerate(classes)
    }
    metrics = {
        "macro_f1": round(float(np.mean(f1_values)), 6),
        "accuracy": round(float((predictions == y).mean()), 6),
    }
    return metrics, confusion, per_class


def evaluate_model(model_id: str, split: str = "validation") -> dict:
    """Evaluate a persisted model; the graph uses validation by default."""
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be 'validation' or 'test'")
    with np.load(_find_model(model_id), allow_pickle=False) as model:
        config = json.loads(str(model["config_json"]))
        manifest_path = Path(str(model["manifest_path"]))
        classes = [str(value) for value in model["classes"].tolist()]
        weights, bias = model["weights"], model["bias"]
        mean, scale = model["mean"], model["scale"]
        training_loss = [float(value) for value in model["training_loss"].tolist()]
        class_counts = model["class_counts"]
    splits, manifest_classes = _load_manifest(manifest_path)
    if manifest_classes != classes:
        raise ValueError("Manifest class order changed after model training")
    if not splits[split]:
        raise ValueError(f"Manifest has no samples in {split!r}")
    x, y, paths = _load_xy(splits[split], classes, config, train=False)
    probs = _softmax(((x - mean) / scale) @ weights + bias)
    predictions = probs.argmax(axis=1)
    threshold = float(config.get("confidence_threshold", 0.0) or 0.0)
    if threshold > 0:
        low_confidence = probs.max(axis=1) < threshold
        predictions[low_confidence] = int(np.argmax(class_counts))
    metrics, confusion, per_class = _metrics(y, predictions, classes)
    validation_loss = float(np.mean(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-8, 1.0))))
    wrong = np.flatnonzero(predictions != y)
    wrong = wrong[np.argsort(probs[wrong, predictions[wrong]])[::-1]][:20]
    misclassified = [
        {
            "image": paths[int(index)],
            "true_class": classes[int(y[index])],
            "predicted_class": classes[int(predictions[index])],
            "confidence": round(float(probs[index, predictions[index]]), 6),
        }
        for index in wrong
    ]
    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "per_class_metrics": per_class,
        "training_history": {
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        },
        "misclassified_examples": misclassified,
        "evaluation_split": split,
    }


def apply_experiment(current_config: dict, proposal: dict) -> dict:
    """Apply only an allow-listed scientist action to a copied config."""
    action = str(proposal.get("next_action") or "")
    if action not in ALLOWED_ACTIONS:
        raise ProposalValidationError(
            f"Invalid next_action={action!r}. Allowed: {sorted(ALLOWED_ACTIONS)}"
        )
    params = dict(proposal.get("parameters") or {})
    updated = deepcopy(current_config)
    mappings = {
        "change_model": ("model_family", "model_family", "efficientnet_b0"),
        "change_learning_rate": ("learning_rate", "learning_rate", 0.0003),
        "change_batch_size": ("batch_size", "batch_size", 64),
        "change_image_size": ("image_size", "image_size", 256),
        "change_augmentation": ("augmentation", "augmentation", "strong"),
        "change_confidence_threshold": ("confidence_threshold", "threshold", 0.4),
    }
    if action in mappings:
        config_key, parameter_key, fallback = mappings[action]
        aliases = {
            "model_family": "model",
            "threshold": "confidence_threshold",
        }
        value = params.get(parameter_key)
        if value is None and parameter_key in aliases:
            value = params.get(aliases[parameter_key])
        updated[config_key] = value if value is not None else params.get("value", fallback)
    elif action == "change_sampler":
        updated["sampler"] = params.get("strategy", params.get("sampler", "weighted"))
    elif action == "change_class_weights":
        updated["class_weights"] = params.get(
            "weights", params.get("class_weights", params.get("value", "balanced"))
        )
    elif action == "increase_epochs":
        if "epochs" in params:
            updated["epochs"] = int(params["epochs"])
        else:
            updated["epochs"] = int(updated.get("epochs", 3)) + int(
                params.get("additional_epochs", 2)
            )
    updated["_last_action"] = action
    updated["_last_parameters"] = params
    return updated


def evaluate_test_model(model_id: str) -> dict:
    """Explicit final-test helper; never used by the optimization graph."""
    return evaluate_model(model_id, split="test")


def get_dataset_summary(config: dict) -> dict:
    """Return manifest-derived facts for the graph/scientist (never hard-coded)."""
    manifest_path = _manifest_path(config)
    splits, classes = _load_manifest(manifest_path)
    train_distribution = {label: 0 for label in classes}
    for record in splits["train"]:
        train_distribution[record["label"]] += 1
    distribution = {label: 0 for label in classes}
    for records in splits.values():
        for record in records:
            distribution[record["label"]] += 1
    positive_counts = [count for count in train_distribution.values() if count]
    imbalance = (
        max(positive_counts) / min(positive_counts) if positive_counts else 0.0
    )
    minority = min(train_distribution, key=train_distribution.get) if train_distribution else ""
    generated = json.loads(manifest_path.read_text(encoding="utf-8")).get(
        "dataset_summary", {}
    )
    return {
        **generated,
        "total_samples": sum(len(records) for records in splits.values()),
        "classes": classes,
        "class_distribution": distribution,
        "train_class_distribution": train_distribution,
        "train_samples": len(splits["train"]),
        "validation_samples": len(splits["validation"]),
        "test_samples": len(splits["test"]),
        "class_imbalance_ratio": round(float(imbalance), 6),
        "minority_class": minority,
        "manifest_path": str(manifest_path),
    }


def evaluate_test(config: dict) -> dict:
    """Retrain the selected config on train only, then evaluate untouched test.

    This intentionally does not merge validation data into training: doing so
    keeps the final operation compatible with the exact split contract used
    during autonomous optimization and avoids any accidental test leakage.
    """
    trained = train_model(config)
    result = evaluate_test_model(trained["model_id"])
    result["model_id"] = trained["model_id"]
    return result
