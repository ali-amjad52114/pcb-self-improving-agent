"""The autonomous loop, driven end to end by Fireworks.

    train -> evaluate -> retrieve memory -> diagnose -> propose (tool call)
          -> apply -> retrain -> evaluate -> critique -> store lesson -> repeat

This is the "first integration milestone" from the 3-person plan, with the fake
Fireworks piece replaced by the real thing. MongoDB is stubbed by
`InMemoryLessonStore` and training by `ml.simulator.Trainer`; both expose the
frozen interfaces, so Person 1 and Person 3 can swap in Atlas and LangGraph
without touching anything in `pcb_agent.fireworks`.

Run A (cold start), then Run B (experienced, reading Run A's lessons):

    python -m pcb_agent.demo.fireworks_loop --run a --budget 5
    python -m pcb_agent.demo.fireworks_loop --run b --budget 5

Run B loads the lesson file Run A wrote. The claim being tested is that it
reaches the target in fewer experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcb_agent.fireworks import (
    ActionHistory,
    FireworksClient,
    FireworksEmbedder,
    FireworksError,
    UsageLedger,
    actions_to_withdraw,
    apply_experiment,
    critique_result,
    diagnose,
    propose_experiment,
    retrieve_memories,
    to_lesson_document,
)
from pcb_agent.fireworks.tools import DEFAULT_TRAINING_CONFIG
from pcb_agent.ml.simulator import TASK_A, TASK_B, SimulatedTask, Trainer

from .memory_store import InMemoryLessonStore

log = logging.getLogger("pcb_agent.demo")

DEFAULT_STORE = Path(".omc/fireworks_demo_lessons.json")


@dataclass
class RunResult:
    run_id: str
    task: str
    baseline_macro_f1: float
    best_macro_f1: float
    experiments_used: int
    target_reached: bool
    target: float
    trajectory: list[float] = field(default_factory=list)
    memories_seen: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        status = "reached" if self.target_reached else "NOT reached"
        arrow = " -> ".join(f"{v:.3f}" for v in self.trajectory)
        return (
            f"[{self.run_id}] {self.task}: target {self.target:.2f} {status} "
            f"in {self.experiments_used} experiment(s)\n"
            f"         {arrow}\n"
            f"         memories retrieved: {self.memories_seen}"
        )


def run_loop(
    client: FireworksClient,
    store: InMemoryLessonStore,
    task: SimulatedTask,
    *,
    run_id: str,
    target: float = 0.82,
    budget: int = 5,
    top_k: int = 4,
    verbose: bool = True,
) -> RunResult:
    """One full agent run against `task`."""
    trainer = Trainer(task)
    config = dict(DEFAULT_TRAINING_CONFIG)
    history: list[dict[str, Any]] = []
    action_history = ActionHistory()
    memories_seen = 0

    # --- baseline -----------------------------------------------------------
    trained = trainer.train_model(config)
    metrics = trainer.evaluate_model(trained["model_id"])
    baseline = float(metrics["macro_f1"])
    best = baseline
    trajectory = [baseline]
    _say(verbose, f"\n[{run_id}] baseline macro F1 = {baseline:.3f}")

    for iteration in range(1, budget + 1):
        if best >= target:
            break

        state = {
            "run_id": run_id,
            "iteration": iteration,
            "dataset_summary": task.dataset_summary(),
            "current_config": config,
            "current_metrics": _headline(metrics),
            "confusion_matrix": metrics["confusion_matrix"],
            "experiment_history": history,
        }

        # --- retrieve memory (vector search, then Fireworks rerank) ---------
        probe = _memory_probe(state, history)
        candidates = store.retrieve_similar_lessons(probe, k=top_k * 3)
        memories = retrieve_memories(client, probe, candidates, top_k=top_k)
        memories_seen += len(memories)
        _say(verbose, f"[{run_id}] exp {iteration}: retrieved {len(memories)} "
                      f"relevant lesson(s) from {len(candidates)} candidate(s)")

        # --- diagnose -------------------------------------------------------
        diagnosis = diagnose(client, state, memories)
        signature = diagnosis.get("failure_signature") or {}
        _say(verbose, f"[{run_id}]   diagnosis: {diagnosis['diagnosis']}")
        _say(verbose, f"[{run_id}]   confidence: {diagnosis.get('confidence', 0):.2f}"
                      f"  mode: {signature.get('failure_mode')}")

        # --- narrow the action space using memory ---------------------------
        failure_mode = signature.get("failure_mode")
        withdrawn = set(actions_to_withdraw(memories, failure_mode))
        withdrawn |= set(action_history.failed_actions(failure_mode or ""))
        allowed = [a for a in action_history.allowed_actions(failure_mode or "")
                   if a not in withdrawn]
        if len(allowed) < 3:
            allowed = None  # never corner the model into a single option
        if withdrawn:
            _say(verbose, f"[{run_id}]   withdrew {sorted(withdrawn)} based on memory")

        # --- propose (forced tool call) --------------------------------------
        proposal = propose_experiment(
            client, state, diagnosis, memories=memories, allowed_actions=allowed
        )
        _say(verbose, f"[{run_id}]   action: {proposal['next_action']}"
                      f"({_compact(proposal['parameters'])})")
        if proposal.get("second_opinion_requested"):
            _say(verbose, f"[{run_id}]   [would escalate to the OpenRouter judge here]")

        # --- execute ---------------------------------------------------------
        before_metrics = _headline(metrics)
        config = apply_experiment(config, proposal)
        trained = trainer.train_model(config)
        metrics = trainer.evaluate_model(trained["model_id"])
        after_metrics = _headline(metrics)
        delta = after_metrics["macro_f1"] - before_metrics["macro_f1"]
        _say(verbose, f"[{run_id}]   macro F1 {before_metrics['macro_f1']:.3f} -> "
                      f"{after_metrics['macro_f1']:.3f} ({delta:+.3f})")

        # --- critique + store -------------------------------------------------
        critique = critique_result(
            client, before_metrics, after_metrics, proposal,
            diagnosis=diagnosis, dataset_summary=task.dataset_summary(),
        )
        experiment_id = store.store_experiment(
            {
                "run_id": run_id,
                "iteration": iteration,
                "dataset_stats": task.dataset_summary(),
                "model_config": config,
                "metrics": after_metrics,
                "failure_signature": signature,
                "intervention": proposal,
                "outcome_delta": critique.get("measured_delta", {}),
                "verdict": critique["verdict"],
                "status": "complete",
            }
        )
        lesson_doc = to_lesson_document(
            critique, experiment_id=experiment_id, run_id=run_id,
            failure_signature=signature,
        )
        lesson_doc["action"] = proposal["next_action"]
        lesson_id = store.store_lesson(lesson_doc)
        _say(verbose, f"[{run_id}]   verdict: {critique['verdict']}  "
                      f"lesson {lesson_id}: {lesson_doc['lesson'][:90]}")

        helped = critique["verdict"] == "improved"
        action_history.record(failure_mode or "unknown", proposal["next_action"],
                              helped=helped)
        history.append(
            {
                "action": proposal["next_action"],
                "parameters": proposal["parameters"],
                "before_macro_f1": before_metrics["macro_f1"],
                "after_macro_f1": after_metrics["macro_f1"],
                "delta": round(delta, 4),
                "verdict": critique["verdict"],
            }
        )
        trajectory.append(after_metrics["macro_f1"])
        best = max(best, after_metrics["macro_f1"])

    return RunResult(
        run_id=run_id,
        task=task.name,
        baseline_macro_f1=baseline,
        best_macro_f1=best,
        experiments_used=len(history),
        target_reached=best >= target,
        target=target,
        trajectory=trajectory,
        memories_seen=memories_seen,
        usage=client.ledger.totals(),
        history=history,
    )


# ------------------------------------------------------------------ helpers


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    """Metrics without the confusion matrix, for the critic and history."""
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "per_class": metrics["per_class"],
    }


def _memory_probe(state: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """The query text used against vector search before we have a diagnosis.

    Built from the worst classes rather than a generic string, so the very
    first retrieval of a run is already targeted.
    """
    per_class = state["current_metrics"]["per_class"]
    worst = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:2]
    parts = [
        f"PCB defect classifier macro F1 {state['current_metrics']['macro_f1']:.2f}",
        "weakest classes: " + ", ".join(
            f"{cls} (f1 {m['f1']:.2f}, recall {m['recall']:.2f}, support {m['support']})"
            for cls, m in worst
        ),
        f"class imbalance ratio {state['dataset_summary']['imbalance_ratio']}",
    ]
    if history:
        parts.append("already tried: " + ", ".join(h["action"] for h in history))
    return "; ".join(parts)


def _compact(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in params.items() if k != "reason")


def _say(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Fireworks-driven self-improving loop."
    )
    parser.add_argument("--run", choices=["a", "b", "both"], default="a",
                        help="a = cold start, b = experienced (reads a's lessons)")
    parser.add_argument("--target", type=float, default=0.82, help="target macro F1")
    parser.add_argument("--budget", type=int, default=5, help="max experiments per run")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE,
                        help="lesson store JSON path")
    parser.add_argument("--fresh", action="store_true",
                        help="clear the lesson store first (forces a true cold start)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv()

    try:
        client = FireworksClient(ledger=UsageLedger(run_id=f"demo_{args.run}"))
    except FireworksError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    embedder = FireworksEmbedder(client)
    store = InMemoryLessonStore(embedder, path=args.store)
    if args.fresh:
        store.clear()
        print(f"cleared lesson store at {args.store}")

    runs = ["a", "b"] if args.run == "both" else [args.run]
    results: list[RunResult] = []

    try:
        for name in runs:
            task = TASK_A if name == "a" else TASK_B
            if name == "b" and not store.lessons:
                print("warning: no lessons stored. Run A first, or Run B is also "
                      "a cold start and the comparison is meaningless.",
                      file=sys.stderr)
            results.append(
                run_loop(
                    client, store, task,
                    run_id=f"run_{name}", target=args.target, budget=args.budget,
                    verbose=not args.quiet,
                )
            )
    except FireworksError as exc:
        print(f"\nfireworks error: {exc}", file=sys.stderr)
        print(client.ledger.summary_line(), file=sys.stderr)
        return 1
    finally:
        client.close()

    if args.json:
        print(json.dumps([vars(r) for r in results], indent=2, default=str))
    else:
        print("\n" + "=" * 70)
        for result in results:
            print(result.summary())
        if len(results) == 2:
            print("-" * 70)
            print(_comparison(results[0], results[1]))
        print("-" * 70)
        print(client.ledger.summary_line())
        print(f"memory store: {store.stats()}")

    return 0


def _comparison(cold: RunResult, experienced: RunResult) -> str:
    if not (cold.target_reached and experienced.target_reached):
        return ("Inconclusive: at least one run did not reach the target within "
                "budget. Raise --budget or lower --target.")
    saved = cold.experiments_used - experienced.experiments_used
    verdict = (
        f"Experienced agent used {saved} fewer experiment(s)."
        if saved > 0 else
        "No improvement from memory this time. Check that Run B retrieved "
        "lessons at all, and that the relevance floor is not filtering everything."
    )
    return (
        f"Cold start:  {cold.experiments_used} experiments to {cold.target:.2f}\n"
        f"Experienced: {experienced.experiments_used} experiments "
        f"({experienced.memories_seen} memories retrieved)\n"
        f"{verdict}"
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
