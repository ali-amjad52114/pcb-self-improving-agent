"""Rich console traces for hackathon demos."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console

from pcb_agent.contracts import metric_value
from pcb_agent.memory_query import summarize_lessons_for_console
from pcb_agent.state import AgentState

# Force UTF-8 friendly output on Windows terminals when possible.
console = Console(force_terminal=True, soft_wrap=True)


def _safe_print(text: str) -> None:
    """Print text without crashing on legacy Windows code pages."""
    try:
        console.print(text)
    except UnicodeEncodeError:
        ascii_text = (
            text.replace("╔", "+")
            .replace("╗", "+")
            .replace("╚", "+")
            .replace("╝", "+")
            .replace("═", "=")
            .replace("║", "|")
            .replace("─", "-")
            .replace("✓", "+")
            .replace("✗", "x")
            .replace("→", "->")
        )
        try:
            sys.stdout.write(ascii_text + "\n")
            sys.stdout.flush()
        except Exception:
            sys.stdout.buffer.write((ascii_text + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()


def print_run_banner(run_id: str) -> None:
    _safe_print(
        f"+==========================================================+\n"
        f"| PCB AUTONOMOUS SCIENTIST // RUN {run_id:<24} |\n"
        f"+==========================================================+"
    )


def print_baseline(state: AgentState) -> None:
    metric = state.get("target_metric") or "macro_f1"
    current = metric_value(state.get("baseline_metrics"), metric)
    target = float(state.get("target_value") or 0.0)
    budget = int(state.get("experiment_budget") or 0)
    _safe_print("\nBASELINE")
    _safe_print(f"macro F1          {current:.3f}")
    _safe_print(f"target            {target:.3f}")
    _safe_print(f"experiment budget {budget}")
    _safe_print(f"[BASELINE] {metric}={current:.3f} target={target:.3f}")


def print_iteration_header(iteration: int) -> None:
    _safe_print(f"\n-------------------- ITERATION {iteration} --------------------\n")


def print_memory(query: str, lessons: list[dict[str, Any]]) -> None:
    short_query = query if len(query) < 80 else query[:77] + "..."
    _safe_print("MEMORY")
    _safe_print(f"query      {short_query}")
    _safe_print(f"retrieved  {len(lessons)} lessons")
    _safe_print(f'[MEMORY] query="{short_query}"')
    _safe_print(f"[MEMORY] retrieved={len(lessons)}")
    rows = summarize_lessons_for_console(lessons)
    if rows:
        _safe_print("\nREUSED EXPERIENCE")
        for helped, text in rows[:5]:
            mark = "+" if helped else "x"
            _safe_print(f"{mark} {text}")


def print_diagnosis(diagnosis: dict[str, Any]) -> None:
    summary = diagnosis.get("summary") or ""
    confidence = float(diagnosis.get("confidence") or 0.0)
    _safe_print("\nSCIENTIST")
    _safe_print(f"diagnosis   {summary}")
    _safe_print(f"confidence  {confidence:.2f}")
    _safe_print(f'[SCIENTIST] diagnosis="{summary}"')
    _safe_print(f"[SCIENTIST] confidence={confidence:.2f}")


def print_experiment(proposal: dict[str, Any], *, memory_changed: bool = False) -> None:
    action = proposal.get("next_action")
    params = proposal.get("parameters") or {}
    _safe_print("\nEXPERIMENT")
    _safe_print(f"action      {action}")
    _safe_print(f"parameters  {params}")
    if memory_changed:
        _safe_print("\nWHY")
        _safe_print("Past experience changed the next experiment.")


def print_judge_skipped() -> None:
    _safe_print("\nJUDGE")
    _safe_print("skipped     scientist confidence sufficient")
    _safe_print("[JUDGE] skipped — scientist confidence sufficient")


def print_judge_result(reason: str, opinion: dict[str, Any]) -> None:
    _safe_print("\nJUDGE")
    _safe_print(f"triggered   reason={reason}")
    _safe_print(
        f"verdict     {opinion.get('verdict')} "
        f"confidence={float(opinion.get('confidence') or 0):.2f} "
        f"recommendation={opinion.get('recommendation')}"
    )
    _safe_print(f"[JUDGE] triggered reason={reason}")
    _safe_print(
        f"[JUDGE] verdict={opinion.get('verdict')} "
        f"confidence={float(opinion.get('confidence') or 0):.2f} "
        f"recommendation={opinion.get('recommendation')}"
    )


def print_result(
    metric: str,
    before: float,
    after: float,
    delta: float,
) -> None:
    sign = "+" if delta >= 0 else ""
    _safe_print("\nRESULT")
    _safe_print(f"macro F1    {before:.3f} -> {after:.3f}")
    _safe_print(f"delta       {sign}{delta:.3f}")
    _safe_print(
        f"[RESULT] {metric} {before:.3f} -> {after:.3f}  delta={sign}{delta:.3f}"
    )


def print_memory_write(experiment_id: str, lesson_ok: bool) -> None:
    _safe_print("\nMEMORY WRITE")
    _safe_print(f"experiment  {experiment_id}")
    _safe_print(f"lesson      {'stored' if lesson_ok else 'FAILED (see warnings)'}")


def print_checkpoint_resumed(run_id: str) -> None:
    _safe_print(
        f"[CHECKPOINT] resumed run {run_id} from persisted LangGraph state"
    )


def print_interrupted(run_id: str) -> None:
    _safe_print(
        "\nRUN INTERRUPTED\n\n"
        f"Checkpoint preserved for run:\n{run_id}\n\n"
        "Resume with:\n\n"
        f"uv run python -m pcb_agent.runner resume --run-id {run_id}\n"
    )


def print_run_complete(
    state: AgentState,
    *,
    checkpoint_backend: str,
    fake_mode: bool = True,
) -> None:
    metric = state.get("target_metric") or "macro_f1"
    baseline = metric_value(state.get("baseline_metrics"), metric)
    best = metric_value(state.get("best_metrics"), metric)
    history = state.get("experiment_history") or []
    lessons_used = sum(len(h.get("memory_used") or []) for h in history)
    budget = int(state.get("experiment_budget") or 0)
    _safe_print(
        "\n+================ RUN COMPLETE =================+\n"
        f"Run                  {state.get('run_id')}\n"
        f"Stop reason           {state.get('stop_reason')}\n"
        f"Baseline macro F1     {baseline:.3f}\n"
        f"Best macro F1         {best:.3f}\n"
        f"Experiments used      {len(history)} / {budget}\n"
        f"Memory lessons used   {lessons_used}\n"
        f"OpenRouter calls      {int(state.get('openrouter_calls') or 0)}\n"
        f"Checkpoint backend    {checkpoint_backend}\n"
        "+================================================+"
    )
    if fake_mode:
        _safe_print("\n[NOTE] Metrics produced by FakeML adapters (demo numbers).")


def print_cold_warm_comparison(cold_exps: int, warm_exps: int, target: float) -> None:
    _safe_print(
        f"\nCOLD START       {cold_exps} experiments -> {target:.2f}\n"
        f"WITH EXPERIENCE  {warm_exps} experiments -> {target:.2f}\n\n"
        f"EXPERIMENTS SAVED: {max(0, cold_exps - warm_exps)}\n\n"
        "Persistent experience saved experiments.\n"
        "MongoDB memory changed what the agent did next."
    )
