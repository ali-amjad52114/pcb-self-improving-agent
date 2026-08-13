"""Deterministic structured generator used by tests and the standalone demo."""

from __future__ import annotations

from typing import Any

from fireworks.schemas import ExperimentCritique, ExperimentProposal


class MockFireworksClient:
    def generate(self, *, system_prompt, payload, response_model, schema_name):
        if response_model is ExperimentCritique:
            return self._critique(payload)
        if response_model is ExperimentProposal:
            return self._proposal(payload)
        raise AssertionError(f"Unexpected response model: {response_model}")

    def _proposal(self, payload: dict[str, Any]) -> ExperimentProposal:
        metrics = payload.get("metrics") or {}
        per_class = payload.get("per_class_metrics") or {}
        dataset = payload.get("dataset_summary") or {}
        history = payload.get("recent_experiments") or []
        memories = payload.get("retrieved_memories") or []
        text = str({"metrics": metrics, "per_class": per_class, "dataset": dataset}).lower()

        failed_actions = {
            str(item.get("action") or item.get("next_action"))
            for item in history
            if str(item.get("verdict") or item.get("result")).lower()
            in {"failed", "no improvement", "worse"}
        }
        useful_memory = next(
            (
                item
                for item in memories
                if "weighted sampl" in str(item).lower()
                and any(word in str(item).lower() for word in ("improv", "+", "0.74"))
            ),
            None,
        )
        imbalance = any(
            token in text
            for token in ("fewer training", "imbalance", "0.31", "minority")
        )
        overfitting = (
            float(metrics.get("training_accuracy", 0))
            - float(metrics.get("validation_accuracy", 0))
            > 0.15
            or "validation loss increasing" in text
        )

        if overfitting:
            return ExperimentProposal(
                diagnosis="Training performance greatly exceeds validation performance.",
                hypothesis="The classifier is overfitting and needs stronger regularization.",
                action="change_augmentation",
                parameters={"augmentation": "strong"},
                reasoning_summary="Stronger augmentation directly tests whether broader input variation improves validation generalization.",
                confidence=0.9,
                memory_used=[],
            )
        if imbalance and "change_sampler" not in failed_actions:
            memory_ids = [str(useful_memory["memory_id"])] if useful_memory else []
            return ExperimentProposal(
                diagnosis="The minority open-circuit class has much lower recall.",
                hypothesis="Class imbalance is limiting minority-class learning.",
                action="change_sampler",
                parameters={"sampler": "weighted"},
                reasoning_summary="Weighted sampling directly increases exposure to the underrepresented class.",
                confidence=0.92 if useful_memory else 0.86,
                memory_used=memory_ids,
            )
        return ExperimentProposal(
            diagnosis="The current configuration is not improving validation performance.",
            hypothesis="Balanced loss contributions may improve minority performance without increasing model capacity.",
            action="change_class_weights",
            parameters={"class_weights": True},
            reasoning_summary="This avoids repeating failed model scaling and tests a class-specific intervention.",
            confidence=0.76,
            memory_used=[],
        )

    def _critique(self, payload: dict[str, Any]) -> ExperimentCritique:
        before = payload["before"].get("metrics", payload["before"])
        after = payload["after"].get("metrics", payload["after"])
        deltas = {
            key: round(float(after[key]) - float(value), 10)
            for key, value in before.items()
            if key in after and isinstance(value, (int, float))
        }
        f1_delta = deltas.get("macro_f1", 0.0)
        verdict = "successful" if f1_delta > 0.01 else "failed"
        return ExperimentCritique(
            verdict=verdict,
            metric_delta=deltas,
            hypothesis_supported=f1_delta > 0.01,
            analysis="Macro F1 and minority recall improved without an observed overall regression.",
            lesson="Weighted sampling improved minority-class recall under severe class imbalance.",
            confidence=0.91,
        )
