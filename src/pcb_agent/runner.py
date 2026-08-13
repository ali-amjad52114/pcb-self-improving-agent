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
) -> AgentState:
    return {
        "run_id": run_id,
        "iteration": 0,
        "experiment_budget": settings.experiment_budget,
        "target_metric": settings.target_metric,
        "target_value": settings.target_value,
        "dataset_summary": {
            "class_imbalance_ratio": 5.4,
            "minority_class": "open_circuit",
        },
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
        "openrouter_call_budget": settings.openrouter_call_budget,
        "openrouter_ledger": [],
        "judge_skipped": False,
        "judge_skip_reason": "",
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
) -> None:
    """Start a new experiment campaign (thread_id == run_id)."""
    try:
        settings, deps, _checkpointer, graph = _build_runtime()
    except (CheckpointError, IntegrationError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    baseline = _load_baseline_config(config)
    initial = _initial_state(run_id=run_id, settings=settings, config=baseline)
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


@app.command("judge-smoke")
def judge_smoke(
    fixture: Path = typer.Option(
        Path("fixtures/judge_context.json"),
        "--fixture",
        exists=False,
        dir_okay=False,
        help="JSON context for a single second_opinion call",
    ),
) -> None:
    """Smoke-test OpenRouter judge without running the full ML loop.

    Uses settings from .env. With OPENROUTER_ENABLED=false this exercises FakeJudge.
    With OPENROUTER_ENABLED=true and keys set, this makes a live panel call.
    """
    settings = get_settings()
    from pcb_agent.integrations.openrouter_judge import build_judge

    if not fixture.exists():
        typer.secho(f"Fixture not found: {fixture}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    context = json.loads(fixture.read_text(encoding="utf-8"))
    context["_openrouter_calls"] = 0
    context["_openrouter_budget"] = settings.openrouter_call_budget

    judge = build_judge(settings)
    typer.echo(
        f"judge={type(judge).__name__} enabled={settings.openrouter_enabled} "
        f"models={settings.openrouter_models()} budget={settings.openrouter_call_budget}"
    )
    opinion = judge.second_opinion(context)
    console.print_judge_result(
        str(context.get("trigger") or "smoke"),
        opinion,
        calls=int(opinion.get("_calls_made") or getattr(judge, "call_count", 0)),
        budget=settings.openrouter_call_budget,
    )
    typer.echo(json.dumps({k: v for k, v in opinion.items() if not str(k).startswith("_")}, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
