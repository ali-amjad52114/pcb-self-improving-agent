"""Run the Person 1 Atlas memory story end to end.

This is deliberately a thin demo adapter around ``pcb_memory.AgentMemory``.
It does not call the scientist, train a model, or construct a LangGraph graph.

Usage:
    python -m scripts.demo_atlas_memory
    python -m scripts.demo_atlas_memory --query "missing hole recall is poor"
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from pcb_memory.memory import AgentMemory


RUN_ID = "run_person1_atlas_demo"

DEMO_EXPERIENCES = (
    {
        "experiment": {
            "experiment_id": "exp_person1_aggressive_lr",
            "run_id": RUN_ID,
            "iteration": 1,
            "status": "complete",
            "model_config": {"learning_rate": 0.003},
            "metrics": {"macro_f1": 0.55, "open_circuit_recall": 0.27},
            "failure_signature": {
                "summary": "Open-circuit recall collapsed after an aggressive learning-rate increase"
            },
            "intervention": {"type": "learning_rate", "multiplier": 3.0},
            "outcome_delta": {"macro_f1": -0.07},
        },
        "lesson": {
            "lesson_id": "lesson_person1_aggressive_lr",
            "run_id": RUN_ID,
            "failure_summary": "Small open circuits have low recall; a large learning-rate jump destabilized training",
            "intervention": "Increase learning rate to three times the baseline",
            "result": "Macro F1 fell from 0.62 to 0.55; avoid aggressive learning-rate jumps",
            "outcome": "failed",
            "defect_family": "open_circuit",
            "before_f1": 0.62,
            "after_f1": 0.55,
            "delta": -0.07,
            "confidence": 0.90,
        },
    },
    {
        "experiment": {
            "experiment_id": "exp_person1_weighted_crops",
            "run_id": RUN_ID,
            "iteration": 2,
            "status": "complete",
            "model_config": {
                "learning_rate": 0.001,
                "sampler": "class_weighted",
                "crop_size": 384,
            },
            "metrics": {"macro_f1": 0.74, "open_circuit_recall": 0.68},
            "failure_signature": {
                "summary": "Small open circuits remain underrepresented and are lost in coarse crops"
            },
            "intervention": {"type": "weighted_sampler_and_tighter_crops"},
            "outcome_delta": {"macro_f1": 0.19},
        },
        "lesson": {
            "lesson_id": "lesson_person1_weighted_crops",
            "run_id": RUN_ID,
            "failure_summary": "Small open circuits have low recall because of class imbalance and coarse crops",
            "intervention": "Use class-weighted sampling and tighter high-resolution crops",
            "result": "Macro F1 rose from 0.55 to 0.74 and open-circuit recall recovered",
            "outcome": "helped",
            "defect_family": "open_circuit",
            "before_f1": 0.55,
            "after_f1": 0.74,
            "delta": 0.19,
            "confidence": 0.93,
        },
    },
)


def _search_indexes(memory: AgentMemory) -> list[dict[str, Any]]:
    """Return judge-friendly index evidence without dumping full definitions."""
    indexes = []
    for item in memory.lessons.list_search_indexes():
        indexes.append(
            {
                "name": item.get("name"),
                "type": item.get("type", "search"),
                "status": item.get("status", "UNKNOWN"),
                "queryable": item.get("queryable", False),
            }
        )
    return indexes


def _seed(memory: AgentMemory) -> list[dict[str, str]]:
    """Commit linked records once; explicit IDs make the demo idempotent."""
    writes = []
    for experience in DEMO_EXPERIENCES:
        lesson_id = experience["lesson"]["lesson_id"]
        experiment_id = experience["experiment"]["experiment_id"]
        already_present = memory.lessons.count_documents({"lesson_id": lesson_id}) > 0
        already_present = already_present and memory.experiments.count_documents(
            {"experiment_id": experiment_id}
        ) > 0
        if already_present:
            writes.append(
                {
                    "experiment_id": experiment_id,
                    "lesson_id": lesson_id,
                    "write": "reused",
                }
            )
            continue

        committed = memory.commit_experience(
            experience["experiment"], experience["lesson"]
        )
        writes.append({**committed, "write": "transaction_committed"})
    return writes


def run(query: str, *, bootstrap: bool = True) -> dict[str, Any]:
    memory = AgentMemory()
    ping = memory.db.command("ping")
    bootstrap_result = (
        memory.bootstrap(wait_for_search=True, search_timeout_s=60.0)
        if bootstrap
        else "skipped"
    )
    writes = _seed(memory)

    hits = memory.retrieve_similar_lessons(
        query,
        k=3,
        defect_family="open_circuit",
        hybrid=True,
    )
    history = memory.get_run_history(RUN_ID)
    best = memory.best_interventions("open_circuit", limit=3)
    learning_evidence = memory.compare_runs([RUN_ID], target_f1=0.74)
    summary = memory.run_summaries.find_one({"run_id": RUN_ID}, {"_id": 0})

    return {
        "story": "Atlas recalls prior outcomes before the next PCB experiment is chosen",
        "atlas": {
            "ping_ok": ping.get("ok") == 1,
            "database": memory.db.name,
            "collections": sorted(memory.db.list_collection_names()),
            "search_indexes": _search_indexes(memory),
            "bootstrap": bootstrap_result,
        },
        "atomic_experience_writes": writes,
        "retrieval": {
            "query": query,
            "mode": "hybrid vector + lexical (vector fallback is built into memory API)",
            "hits": hits,
        },
        "linked_run_history": history,
        "computed_best_interventions": best,
        "atlas_learning_evidence": learning_evidence,
        "materialized_run_summary": summary,
        "handoff": {
            "person_2_reads": "retrieval.hits as evidence for the next experiment proposal",
            "person_3_calls": "AgentMemory methods from graph nodes; no graph code lives here",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="Small open circuits still have poor recall because the class is rare",
        help="A current model failure to retrieve prior lessons for.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip collection and Atlas Search index creation on later runs.",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.query, bootstrap=not args.skip_bootstrap), default=str, indent=2))


if __name__ == "__main__":
    main()
