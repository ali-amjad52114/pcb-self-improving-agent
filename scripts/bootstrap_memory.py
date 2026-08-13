"""Create Atlas collections/indexes and seed fake lessons for the first integration hour."""

from __future__ import annotations

import json

from pcb_memory.memory import AgentMemory


SEED_LESSONS = [
    {
        "run_id": "run_cold_start_demo",
        "failure_summary": "Small open circuits have low recall and class imbalance",
        "intervention": "weighted sampler + tighter image crops",
        "result": "+11% macro F1; minority open-circuit recall recovered",
        "outcome": "helped",
        "defect_family": "open_circuit",
        "before_f1": 0.67,
        "after_f1": 0.78,
        "delta": 0.11,
        "confidence": 0.91,
    },
    {
        "run_id": "run_cold_start_demo",
        "failure_summary": "Missing-hole class is underrepresented and recall is extremely poor",
        "intervention": "weighted sampling",
        "result": "Minority recall improved; macro F1 0.58 → 0.69",
        "outcome": "helped",
        "defect_family": "missing_hole",
        "before_f1": 0.58,
        "after_f1": 0.69,
        "delta": 0.11,
        "confidence": 0.88,
    },
    {
        "run_id": "run_cold_start_demo",
        "failure_summary": "Increasing learning rate to chase F1 caused training collapse",
        "intervention": "change_learning_rate to 3x baseline",
        "result": "F1 dropped; do not jump LR this aggressively on this dataset",
        "outcome": "failed",
        "defect_family": "open_circuit",
        "before_f1": 0.69,
        "after_f1": 0.52,
        "delta": -0.17,
        "confidence": 0.86,
    },
    {
        "run_id": "run_cold_start_demo",
        "failure_summary": "Small defects remain hard after class-weighting; crops still too coarse",
        "intervention": "increase crop resolution and augmentation",
        "result": "Open-circuit recall rose; macro F1 0.69 → 0.78",
        "outcome": "helped",
        "defect_family": "open_circuit",
        "before_f1": 0.69,
        "after_f1": 0.78,
        "delta": 0.09,
        "confidence": 0.84,
    },
]


def main() -> None:
    memory = AgentMemory()
    info = memory.bootstrap()
    print("bootstrap:", json.dumps(info, default=str))
    ids = []
    for lesson in SEED_LESSONS:
        ids.append(memory.store_lesson(lesson))
        memory.store_experiment(
            {
                "run_id": lesson["run_id"],
                "status": "complete",
                "metrics": {"macro_f1": lesson["after_f1"]},
                "failure_signature": {"text": lesson["failure_summary"]},
                "intervention": {"text": lesson["intervention"]},
                "outcome_delta": {"macro_f1": lesson["delta"]},
            }
        )
    print("seeded lessons:", ids)
    query = "Small open circuits have low recall and class imbalance"
    hits = memory.retrieve_similar_lessons(query, k=3)
    print("retrieve:", json.dumps(hits, default=str, indent=2)[:4000])


if __name__ == "__main__":
    main()
