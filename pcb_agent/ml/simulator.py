"""A stand-in PCB trainer, so the Fireworks loop runs end to end today.

This is deliberately a placeholder. Person 2 owns the real training and
evaluation pipeline; this module exists so the Fireworks scientist, critic,
memory retrieval, and the LangGraph wiring can all be exercised and demoed
before a GPU is involved.

It is not a random number generator dressed up as a model. Each simulated task
has a hidden set of bottlenecks (class imbalance, defect scale, calibration,
capacity), and macro F1 responds only to interventions that address a real
bottleneck. So an agent that reasons correctly measurably outperforms one that
guesses, which is the property the demo needs to be worth anything.

Swap it out by giving `train_model` / `evaluate_model` the same signatures:

    train_model(config: dict) -> dict
    evaluate_model(model_id: str) -> dict
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

PCB_CLASSES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper",
]


@dataclass
class SimulatedTask:
    """A PCB classification problem with a known, hidden set of bottlenecks."""

    name: str
    class_counts: dict[str, int]
    defect_scale_px: int = 18       # typical defect size in source pixels
    label_noise: float = 0.02
    capacity_needed: str = "resnet18"
    seed: int = 7
    ceiling: float = 0.93           # best achievable macro F1 on this task

    @property
    def imbalance_ratio(self) -> float:
        counts = list(self.class_counts.values())
        return max(counts) / max(1, min(counts))

    def dataset_summary(self) -> dict[str, Any]:
        total = sum(self.class_counts.values())
        return {
            "name": self.name,
            "task": "PCB surface defect classification",
            "n_samples": total,
            "n_classes": len(self.class_counts),
            "class_counts": dict(self.class_counts),
            "class_fractions": {
                k: round(v / total, 4) for k, v in self.class_counts.items()
            },
            "imbalance_ratio": round(self.imbalance_ratio, 2),
            "source_resolution": "600x600",
            "notes": "Defect regions are small relative to the board image.",
        }

    def minority_classes(self, threshold: float = 0.08) -> list[str]:
        total = sum(self.class_counts.values())
        return [k for k, v in self.class_counts.items() if v / total < threshold]


#: Run A. Heavy imbalance plus small defects: the cold-start agent has to find
#: both, which is what makes it take several experiments.
TASK_A = SimulatedTask(
    name="pcb_defects_v1",
    class_counts={
        "missing_hole": 1200, "mouse_bite": 1100, "open_circuit": 140,
        "short": 980, "spur": 210, "spurious_copper": 1050,
    },
    defect_scale_px=14,
    capacity_needed="resnet18",
    seed=7,
)

#: Run B. A related task: same failure *modes*, different dataset and class
#: mix. Lessons from Run A transfer; the exact numbers do not.
TASK_B = SimulatedTask(
    name="pcb_defects_v2_smt",
    class_counts={
        "missing_hole": 900, "mouse_bite": 870, "open_circuit": 95,
        "short": 760, "spur": 130, "spurious_copper": 820,
    },
    defect_scale_px=12,
    capacity_needed="resnet18",
    seed=23,
    ceiling=0.91,
)


ARCH_CAPACITY = {
    "resnet18": 1.0, "efficientnet_b0": 1.05, "convnext_tiny": 1.15,
    "resnet50": 1.2, "efficientnet_b3": 1.25, "vit_small_patch16": 1.15,
}


@dataclass
class Trainer:
    """Holds task state across experiments so metrics are reproducible."""

    task: SimulatedTask
    _runs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def train_model(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Train with `config`. Returns `{model_id, config, epochs_run}`."""
        model_id = _config_id(self.task.name, config)
        self._runs[model_id] = {"config": dict(config)}
        return {
            "model_id": model_id,
            "config": dict(config),
            "epochs_run": int(config.get("epochs", 12)),
            "task": self.task.name,
        }

    def evaluate_model(self, model_id: str) -> dict[str, Any]:
        """Evaluate a trained model. Returns metrics plus a confusion matrix."""
        run = self._runs.get(model_id)
        if run is None:
            raise KeyError(f"unknown model_id {model_id!r}; call train_model first")
        return self._metrics(run["config"], model_id)

    # -------------------------------------------------------------- scoring

    def _metrics(self, config: Mapping[str, Any], model_id: str) -> dict[str, Any]:
        task = self.task
        total = sum(task.class_counts.values())
        jitter = _jitter(model_id, task.seed)

        per_class: dict[str, dict[str, float]] = {}
        for cls, count in task.class_counts.items():
            fraction = count / total
            recall = self._class_recall(config, fraction, jitter)
            precision = self._class_precision(config, fraction, jitter)
            f1 = 0.0 if (precision + recall) == 0 else (
                2 * precision * recall / (precision + recall)
            )
            per_class[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": count,
            }

        macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
        accuracy = sum(
            per_class[c]["recall"] * task.class_counts[c] for c in task.class_counts
        ) / total

        return {
            "model_id": model_id,
            "macro_f1": round(min(macro_f1, task.ceiling), 4),
            "accuracy": round(accuracy, 4),
            "per_class": per_class,
            "confusion_matrix": self._confusion(per_class),
            "n_eval_samples": total // 5,
        }

    #: Fraction below which a class counts as a minority for this simulation.
    MINORITY_FRACTION = 0.10
    #: Effective defect size (in model pixels) below which resolution binds.
    RESOLVABLE_PX = 16.0

    def _class_recall(
        self, config: Mapping[str, Any], fraction: float, jitter: float
    ) -> float:
        task = self.task
        base = 0.88

        # Bottleneck 1: class imbalance. Rare classes are under-recalled unless
        # the sampler or the loss weights compensate. Both help, and they
        # compose, but with diminishing returns.
        if fraction < self.MINORITY_FRACTION:
            penalty = 0.65 * (1 - fraction / self.MINORITY_FRACTION)
            sampler = str(config.get("sampler", "random"))
            weights = config.get("class_weights")
            if sampler in ("weighted", "balanced_batch", "oversample_minority"):
                penalty *= 0.35
            if weights in ("balanced", "inverse_frequency") or isinstance(weights, dict):
                penalty *= 0.55
            base -= penalty

        # Bottleneck 2: defect scale. Defects occupy a few pixels at the default
        # input resolution, which caps recall for every class at once. No amount
        # of reweighting recovers detail that was never sampled.
        effective_px = task.defect_scale_px * (int(config.get("image_size", 224)) / 600)
        if effective_px < self.RESOLVABLE_PX:
            base -= 0.42 * (1 - effective_px / self.RESOLVABLE_PX)

        # Bottleneck 3: capacity. Too little hurts a lot; surplus helps a little.
        capacity = ARCH_CAPACITY.get(str(config.get("architecture", "resnet18")), 1.0)
        needed = ARCH_CAPACITY.get(task.capacity_needed, 1.0)
        if capacity < needed:
            base -= 0.35 * (needed - capacity)
        else:
            base += min(0.04, 0.08 * (capacity - needed))

        base += _lr_penalty(config) + _aug_bonus(config, fraction,
                                                 self.MINORITY_FRACTION)
        base -= task.label_noise

        # Lowering the threshold trades precision for recall.
        threshold = float(config.get("confidence_threshold", 0.5))
        base += (0.5 - threshold) * 0.25

        return _clamp(base + jitter, 0.02, 0.99)

    def _class_precision(
        self, config: Mapping[str, Any], fraction: float, jitter: float
    ) -> float:
        threshold = float(config.get("confidence_threshold", 0.5))
        base = 0.88 - (0.5 - threshold) * 0.35
        if fraction < self.MINORITY_FRACTION:
            # Aggressive oversampling buys recall at the cost of precision.
            factor = float(config.get("oversample_factor", 1.0))
            base -= 0.05 * math.log1p(max(0.0, factor - 1.0))
        base -= self.task.label_noise
        return _clamp(base + jitter * 0.5, 0.05, 0.99)

    def _confusion(self, per_class: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
        """A plausible confusion matrix consistent with the per-class recalls.

        Errors are pushed into the visually nearest class, which is what gives
        the scientist a real confusion pair to reason about.
        """
        neighbours = {
            "missing_hole": "spurious_copper", "mouse_bite": "spur",
            "open_circuit": "mouse_bite", "short": "spurious_copper",
            "spur": "mouse_bite", "spurious_copper": "short",
        }
        matrix: dict[str, dict[str, int]] = {}
        for cls, metrics in per_class.items():
            support = int(metrics["support"]) // 5
            correct = int(round(support * float(metrics["recall"])))
            wrong = support - correct
            row = {c: 0 for c in per_class}
            row[cls] = correct
            row[neighbours.get(cls, cls)] += int(wrong * 0.7)
            leftover = wrong - int(wrong * 0.7)
            for other in per_class:
                if other != cls and leftover > 0:
                    row[other] += 1
                    leftover -= 1
            matrix[cls] = row
        return {"labels": list(per_class), "matrix": matrix}


def _lr_penalty(config: Mapping[str, Any]) -> float:
    lr = float(config.get("learning_rate", 3e-4))
    if lr <= 0:
        return -0.5
    # Log-parabola peaked near 3e-4.
    return -0.08 * (math.log10(lr) - math.log10(3e-4)) ** 2


def _aug_bonus(config: Mapping[str, Any], fraction: float,
               minority_fraction: float = 0.10) -> float:
    augs = list(config.get("augmentations") or [])
    strength = float(config.get("aug_strength", 0.3))
    bonus = min(len(augs), 5) * 0.012
    if fraction < minority_fraction and any(
        a in augs for a in ("mixup", "cutmix", "random_erasing")
    ):
        bonus += 0.03
    if strength > 0.7:
        bonus -= (strength - 0.7) * 0.25  # too much augmentation hurts
    return bonus


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _config_id(task_name: str, config: Mapping[str, Any]) -> str:
    payload = repr(sorted((k, repr(v)) for k, v in config.items()))
    digest = hashlib.sha256(f"{task_name}|{payload}".encode()).hexdigest()[:12]
    return f"model_{digest}"


def _jitter(model_id: str, seed: int) -> float:
    """Deterministic per-config noise, so the same config always scores the
    same but neighbouring configs are not identical."""
    digest = hashlib.sha256(f"{seed}:{model_id}".encode()).digest()
    raw = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return (raw - 0.5) * 0.012
