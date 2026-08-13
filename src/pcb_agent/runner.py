"""Typer CLI — start / resume / status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer

from pcb_agent import console
from pcb_agent.checkpoint import get_checkpointer
from pcb_agent.config import Settings, get_settings
from pcb_agent.contracts import CheckpointError, IntegrationError
from pcb_agent.dependencies import build_dependencies
from pcb_agent.graph import build_graph
from pcb_agent.state import AgentState

app = typer.Typer(
    add_completion=False,
    help="PCB self-improving agent — Person 3 LangGraph orchestrator",
    invoke_without_command=False,
)


def _load_baseline_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        default = Path("configs/baseline.json")
        path = default if default.exists() else None
    if path is None:
        return {
            "model_family": "resnet18",
            "learning_rate": 0.001,
            "batch_size": 32,
            "image_size": 224,
            "augmentation": "basic",
            "sampler": "random",
            "class_weights": None,
            "confidence_threshold": 0.5,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_state(
    *,
    run_id: str,
    settings: Settings,
    config: dict[str, Any],
    dataset_summary: dict[str, Any] | None = None,
    memory_mode: str = "memory",
) -> AgentState:
    return {
        "run_id": run_id,
        "iteration": 0,
        "experiment_budget": settings.experiment_budget,
        "target_metric": settings.target_metric,
        "target_value": settings.target_value,
        "dataset_summary": dataset_summary or {},
        "current_config": config,
        "best_config": config,
        "current_model_id": "",
        "baseline_metrics": {},
        "previous_metrics": {},
        "current_metrics": {},
        "best_metrics": {},
        "confusion_matrix": {},
        "per_class_metrics": {},
        "memory_query": "",
        "retrieved_lessons": [],
        "diagnosis": {},
        "proposed_experiment": {},
        "proposal_revision_count": 0,
        "evaluator_opinion": {},
        "evaluator_trigger": "",
        "critique": {},
        "experiment_history": [],
        "consecutive_failures": 0,
        "last_metric_delta": 0.0,
        "last_experiment_helped": False,
        "status": "started",
        "stop_reason": "",
        "warnings": [],
        "pending_experiment_id": "",
        "openrouter_calls": 0,
        "judge_skipped": False,
        "memory_mode": memory_mode,
        "training_history": {},
        "misclassified_examples": [],
        "previous_confusion_matrix": {},
        "previous_per_class_metrics": {},
        "final_test_metrics": {},
    }


def _build_runtime(settings: Settings | None = None):
    settings = settings or get_settings()
    deps = build_dependencies(settings)
    checkpointer = get_checkpointer(settings)
    graph = build_graph(deps, checkpointer, settings)
    return settings, deps, checkpointer, graph


@app.command("start")
def start(
    run_id: str = typer.Option(..., "--run-id", help="Durable run / thread id"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        help="Baseline training config JSON",
    ),
    mode: str = typer.Option(
        "memory",
        "--mode",
        help="Memory retrieval mode: cold or memory",
    ),
) -> None:
    """Start a new experiment campaign (thread_id == run_id)."""
    try:
        settings, deps, _checkpointer, graph = _build_runtime()
    except (CheckpointError, IntegrationError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    mode = mode.lower().strip()
    if mode not in {"cold", "memory"}:
        typer.secho("--mode must be 'cold' or 'memory'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    baseline = _load_baseline_config(config)
    dataset_summary_getter = getattr(deps.ml, "get_dataset_summary", None)
    dataset_summary = (
        dict(dataset_summary_getter(baseline))
        if callable(dataset_summary_getter)
        else {}
    )
    initial = _initial_state(
        run_id=run_id,
        settings=settings,
        config=baseline,
        dataset_summary=dataset_summary,
        memory_mode=mode,
    )
    thread_config = {"configurable": {"thread_id": run_id}}

    console.print_run_banner(run_id)
    backend = settings.checkpointer_backend
    if backend == "mongodb":
        typer.echo(f"[CHECKPOINT] backend=MongoDB db={settings.mongodb_db_name}")
    else:
        typer.echo("[CHECKPOINT] backend=InMemorySaver")

    try:
        final_state = graph.invoke(initial, config=thread_config)
    except KeyboardInterrupt:
        console.print_interrupted(run_id)
        raise typer.Exit(code=130) from None

    test_evaluator = getattr(deps.ml, "evaluate_test", None)
    if callable(test_evaluator) and settings.agent_mode == "integrated":
        try:
            test_result = dict(test_evaluator(dict(final_state.get("best_config") or baseline)))
            final_state["final_test_metrics"] = dict(
                test_result.get("metrics") or test_result
            )
        except Exception as exc:
            warnings = list(final_state.get("warnings") or [])
            warnings.append(f"final_test_evaluation_failed: {exc}")
            final_state["warnings"] = warnings

    console.print_run_complete(
        final_state,  # type: ignore[arg-type]
        checkpoint_backend="MongoDB" if backend == "mongodb" else "memory",
        fake_mode=settings.agent_mode == "fake",
    )


@app.command("resume")
def resume(
    run_id: str = typer.Option(..., "--run-id", help="Existing run / thread id"),
) -> None:
    """Resume a checkpointed campaign. Reuses the same thread_id."""
    try:
        settings, _deps, _checkpointer, graph = _build_runtime()
    except (CheckpointError, IntegrationError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    thread_config = {"configurable": {"thread_id": run_id}}
    console.print_checkpoint_resumed(run_id)

    try:
        final_state = graph.invoke(None, config=thread_config)
    except KeyboardInterrupt:
        console.print_interrupted(run_id)
        raise typer.Exit(code=130) from None

    if final_state:
        console.print_run_complete(
            final_state,  # type: ignore[arg-type]
            checkpoint_backend=(
                "MongoDB" if settings.checkpointer_backend == "mongodb" else "memory"
            ),
            fake_mode=settings.agent_mode == "fake",
        )
    else:
        typer.secho(
            f"No checkpoint found for run_id={run_id}. Start a new run first.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)


@app.command("status")
def status(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Minimal checkpoint status inspection."""
    try:
        settings, _deps, checkpointer, graph = _build_runtime()
    except (CheckpointError, IntegrationError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    thread_config = {"configurable": {"thread_id": run_id}}
    try:
        snap = graph.get_state(thread_config)
    except Exception as exc:
        typer.secho(f"Unable to read checkpoint: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    values = snap.values or {}
    typer.echo(f"run_id={run_id}")
    typer.echo(f"status={values.get('status')}")
    typer.echo(f"iteration={values.get('iteration')}")
    typer.echo(f"stop_reason={values.get('stop_reason')}")
    typer.echo(f"next={snap.next}")
    typer.echo(f"checkpointer={settings.checkpointer_backend}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
