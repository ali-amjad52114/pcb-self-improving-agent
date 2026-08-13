"""Deterministic fake scientist — proposals depend on retrieved lessons."""

from __future__ import annotations

from typing import Any


# Inefficient cold-start exploration order (no useful memory).
COLD_SEQUENCE = [
    "change_learning_rate",
    "change_batch_size",
    "change_sampler",
    "change_class_weights",
]

ACTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "change_learning_rate": {
        "parameters": {"learning_rate": 0.0003},
        "hypothesis": "Lower learning rate may stabilize minority-class learning",
        "expected_effect": "Modest macro F1 gain",
    },
    "change_batch_size": {
        "parameters": {"batch_size": 64},
        "hypothesis": "Larger batches may reduce gradient noise",
        "expected_effect": "Improve accuracy",
    },
    "change_sampler": {
        "parameters": {"strategy": "weighted"},
        "hypothesis": "Weighted sampling should improve minority recall",
        "expected_effect": "Improve macro F1 via open-circuit recall",
    },
    "change_class_weights": {
        "parameters": {"weights": "balanced"},
        "hypothesis": "Class weights compensate for imbalance",
        "expected_effect": "Improve minority recall",
    },
    "change_augmentation": {
        "parameters": {"augmentation": "strong"},
        "hypothesis": "Stronger augmentation improves generalization",
        "expected_effect": "Small macro F1 gain",
    },
    "change_image_size": {
        "parameters": {"image_size": 256},
        "hypothesis": "Higher resolution helps small defects",
        "expected_effect": "Small recall gain on fine defects",
    },
    "change_confidence_threshold": {
        "parameters": {"threshold": 0.4},
        "hypothesis": "Lower threshold recovers missed defects",
        "expected_effect": "Tiny metric change",
    },
    "change_model": {
        "parameters": {"model_family": "efficientnet_b0"},
        "hypothesis": "Stronger backbone may lift overall F1",
        "expected_effect": "Moderate macro F1 gain",
    },
}


def _lesson_text(lesson: dict[str, Any]) -> str:
    return " ".join(
        str(lesson.get(k, ""))
        for k in ("failure_summary", "intervention", "result", "next_action")
    ).lower()


def _helped(lesson: dict[str, Any]) -> bool | None:
    if "helped" in lesson:
        return bool(lesson["helped"])
    result = str(lesson.get("result") or "").lower()
    if result.startswith("-") or "did not" in result or "failed" in result:
        return False
    if result.startswith("+") or "improv" in result:
        return True
    return None


def _intervention_action(lesson: dict[str, Any]) -> str | None:
    if lesson.get("next_action"):
        return str(lesson["next_action"])
    intervention = str(lesson.get("intervention") or "").lower()
    mapping = [
        ("sampler", "change_sampler"),
        ("weighted", "change_sampler"),
        ("class weight", "change_class_weights"),
        ("batch", "change_batch_size"),
        ("learning rate", "change_learning_rate"),
        ("lr ", "change_learning_rate"),
        ("augment", "change_augmentation"),
        ("image size", "change_image_size"),
        ("threshold", "change_confidence_threshold"),
        ("model", "change_model"),
    ]
    for needle, action in mapping:
        if needle in intervention:
            return action
    return None


class FakeReasoning:
    """Person 2 reasoning contract that reacts to memory contents."""

    def diagnose(self, state: dict, memories: list) -> dict:
        per_class = state.get("per_class_metrics") or {}
        oc = per_class.get("open_circuit") or {}
        recall = float(oc.get("recall", 0.41))
        evidence = [
            f"open_circuit recall = {recall:.2f}",
            "class imbalance ratio = 5.4",
        ]
        if memories:
            evidence.append(f"retrieved {len(memories)} prior lessons")
        confidence = 0.82 if recall < 0.7 else 0.7
        # Allow tests / judge policy demos to lower confidence via state.
        if state.get("_force_low_diagnosis_confidence"):
            confidence = 0.4
        return {
            "summary": "Open-circuit recall is poor because minority samples are underrepresented",
            "confidence": confidence,
            "evidence": evidence,
        }

    def propose_experiment(self, state: dict, diagnosis: dict) -> dict:
        all_memories = list(state.get("retrieved_lessons") or [])
        run_id = state.get("run_id")
        # Only prior-run lessons should change strategy (demo: cold vs experienced).
        # Same-run writes must not scramble the in-progress exploration order.
        memories = [
            m
            for m in all_memories
            if not run_id or m.get("run_id") in (None, "",) or m.get("run_id") != run_id
        ]
        history = list(state.get("experiment_history") or [])
        tried = {h.get("proposal", {}).get("next_action") for h in history}

        opinion = state.get("evaluator_opinion") or {}
        revision = int(state.get("proposal_revision_count") or 0)

        preferred: list[str] = []
        avoided: set[str] = set()
        for lesson in memories:
            action = _intervention_action(lesson)
            if not action:
                continue
            outcome = _helped(lesson)
            if outcome is True:
                preferred.append(action)
            elif outcome is False:
                avoided.add(action)

        # Experienced path: prefer known-good minority interventions first.
        # Rank positive lessons by typical impact so warm runs hit target sooner.
        impact_order = [
            "change_sampler",
            "change_class_weights",
            "change_batch_size",
            "change_learning_rate",
            "change_model",
            "change_augmentation",
            "change_image_size",
            "change_confidence_threshold",
        ]
        candidate_order: list[str] = []
        preferred_set = set(preferred)
        for action in impact_order:
            if action in preferred_set and action not in avoided:
                candidate_order.append(action)
        if any("sampler" in _lesson_text(m) and _helped(m) for m in memories):
            if "change_sampler" not in candidate_order and "change_sampler" not in avoided:
                candidate_order.insert(0, "change_sampler")
            elif "change_sampler" in candidate_order:
                candidate_order.remove("change_sampler")
                candidate_order.insert(0, "change_sampler")
        for action in ("change_class_weights", "change_augmentation"):
            if action not in candidate_order and action not in avoided:
                # Prefer class_weights early when sampler already confirmed.
                if action == "change_class_weights" and "change_sampler" in candidate_order:
                    candidate_order.insert(1, action)
                else:
                    candidate_order.append(action)
        for action in COLD_SEQUENCE:
            if action not in candidate_order and action not in avoided:
                candidate_order.append(action)

        # Cold path with no prior-run memory: inefficient sequence.
        if not memories:
            candidate_order = list(COLD_SEQUENCE)

        chosen = None
        for action in candidate_order:
            if action not in tried:
                chosen = action
                break
        if chosen is None:
            chosen = "change_augmentation"

        prev = state.get("proposed_experiment") or {}
        if (
            opinion.get("recommendation") == "reconsider"
            and prev.get("next_action")
            and revision >= 1
        ):
            suggested = str(opinion.get("suggested_action_family") or "")
            if (
                suggested
                and suggested in ACTION_DEFAULTS
                and suggested != prev.get("next_action")
                and suggested not in tried
            ):
                chosen = suggested
            else:
                for action in candidate_order:
                    if action != prev.get("next_action") and action not in tried:
                        chosen = action
                        break

        defaults = ACTION_DEFAULTS[chosen]
        confidence = 0.81
        if state.get("_force_low_proposal_confidence"):
            confidence = 0.4
        memory_confirmed = bool(memories) and (
            chosen in preferred
            or (
                chosen == "change_sampler"
                and any("sampler" in _lesson_text(m) and _helped(m) for m in memories)
            )
        )
        if memory_confirmed:
            confidence = 0.9

        parameters = dict(defaults["parameters"])
        # Signal FakeML that this action is backed by prior positive lessons.
        # Amplifies measured effect without a hidden warm_run boolean.
        if memory_confirmed:
            parameters["memory_confirmed"] = True

        return {
            "diagnosis": diagnosis.get("summary")
            or "Minority classes are underrepresented",
            "hypothesis": defaults["hypothesis"],
            "next_action": chosen,
            "parameters": parameters,
            "expected_effect": defaults["expected_effect"],
            "confidence": confidence,
        }

    def critique_result(self, before: dict, after: dict, action: dict) -> dict:
        before_f1 = float((before or {}).get("macro_f1", 0.0))
        after_f1 = float((after or {}).get("macro_f1", 0.0))
        delta = after_f1 - before_f1
        helped = delta > 1e-6
        intervention = str(action.get("next_action") or "unknown")
        params = action.get("parameters") or {}
        if intervention == "change_sampler":
            label = f"weighted sampler ({params.get('strategy', 'weighted')})"
        elif intervention == "change_batch_size":
            label = f"batch size {params.get('batch_size', '')}".strip()
        elif intervention == "change_learning_rate":
            label = f"learning rate {params.get('learning_rate', '')}".strip()
        elif intervention == "change_class_weights":
            label = "balanced class weights"
        else:
            label = intervention.replace("change_", "").replace("_", " ")

        sign = "+" if delta >= 0 else ""
        return {
            "helped": helped,
            "reason": (
                "Macro F1 improved and minority recall increased"
                if helped
                else "Macro F1 did not improve for this intervention"
            ),
            "lesson": {
                "failure_summary": "Poor recall on minority open-circuit class",
                "intervention": label,
                "result": f"{sign}{delta:.2f} macro F1",
                "confidence": 0.89 if helped else 0.87,
                "next_action": intervention,
                "helped": helped,
            },
        }
