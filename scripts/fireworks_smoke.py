#!/usr/bin/env python3
"""Verify every Fireworks capability this project depends on, against the live API.

Run this first, before debugging the agent. It tells you which capabilities
work with your key today, and it checks the configured model roster against the
live catalogue, which is the failure that wastes the most time (a model id that
was serverless last month now 404s).

    python scripts/fireworks_smoke.py             # everything
    python scripts/fireworks_smoke.py --models    # registry check only, 1 call
    python scripts/fireworks_smoke.py --only chat,tools

Exit code is 0 only if every selected check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcb_agent.fireworks import (  # noqa: E402
    FireworksClient,
    FireworksError,
    FireworksSettings,
    Role,
    UsageLedger,
    complete_json,
    parse_tool_calls,
    tool_definitions,
    validate_registry,
)
from pcb_agent.fireworks.embeddings import FireworksEmbedder, cosine_similarity  # noqa: E402
from pcb_agent.fireworks.schemas import DIAGNOSIS_SCHEMA  # noqa: E402
from pcb_agent.fireworks.structured import (  # noqa: E402
    VERDICT_GRAMMAR,
    grammar_response_format,
)

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

SAMPLE_METRICS = {
    "macro_f1": 0.58,
    "per_class": {
        "open_circuit": {"precision": 0.71, "recall": 0.32, "f1": 0.44, "support": 140},
        "spur": {"precision": 0.66, "recall": 0.41, "f1": 0.51, "support": 210},
        "short": {"precision": 0.88, "recall": 0.91, "f1": 0.89, "support": 980},
    },
}


def check_models(client: FireworksClient) -> str:
    catalogue = client.list_models()
    result = validate_registry(catalogue)
    print(DIM + result.report() + RESET)
    if not result.ok:
        raise RuntimeError(f"{len(result.missing)} configured model(s) are unavailable")
    return f"{len(catalogue)} models visible, all {len(result.available)} roles resolve"


def check_chat(client: FireworksClient) -> str:
    payload = client.chat(
        [{"role": "user", "content": "Reply with exactly: pong"}],
        role=Role.FAST, max_tokens=16, temperature=0.0, purpose="smoke.chat",
    )
    text = (payload["choices"][0]["message"]["content"] or "").strip()
    return f"replied {text[:40]!r}"


def check_streaming(client: FireworksClient) -> str:
    pieces = list(
        client.stream_text(
            [{"role": "user", "content": "Count from 1 to 5, space separated."}],
            role=Role.FAST, max_tokens=48, temperature=0.0, purpose="smoke.stream",
        )
    )
    if not pieces:
        raise RuntimeError("stream produced no content deltas")
    return f"{len(pieces)} deltas, {''.join(pieces).strip()[:40]!r}"


def check_structured(client: FireworksClient) -> str:
    result = complete_json(
        client,
        name="Diagnosis",
        schema=DIAGNOSIS_SCHEMA,
        messages=[
            {"role": "system", "content": "You are an ML scientist."},
            {"role": "user", "content":
                "A PCB defect classifier scores these metrics. Diagnose the "
                "main problem.\n" + json.dumps(SAMPLE_METRICS, indent=2)},
        ],
        role=Role.SCIENTIST,
        purpose="smoke.structured",
    )
    mode = (result.get("failure_signature") or {}).get("failure_mode")
    return f"valid Diagnosis, failure_mode={mode!r}, confidence={result.get('confidence')}"


def check_tools(client: FireworksClient) -> str:
    payload = client.chat(
        [
            {"role": "system", "content": "Choose exactly one experiment tool."},
            {"role": "user", "content":
                "open_circuit recall is 0.32 with only 140 of 3680 samples. "
                "Choose the single best intervention."},
        ],
        role=Role.SCIENTIST,
        tools=tool_definitions(),
        tool_choice="required",
        parallel_tool_calls=False,
        temperature=0.1,
        purpose="smoke.tools",
    )
    calls = parse_tool_calls(payload)
    call = calls[0]
    return f"chose {call.name}({', '.join(k for k in call.arguments if k != 'reason')})"


def check_grammar(client: FireworksClient) -> str:
    payload = client.chat(
        [{"role": "user", "content":
            "Macro F1 went from 0.58 to 0.71. One word verdict."}],
        role=Role.FAST,
        response_format=grammar_response_format(VERDICT_GRAMMAR),
        max_tokens=16,
        temperature=0.0,
        purpose="smoke.grammar",
    )
    text = (payload["choices"][0]["message"]["content"] or "").strip()
    allowed = {"improved", "regressed", "no_change", "mixed"}
    if text not in allowed:
        raise RuntimeError(f"grammar mode returned {text!r}, expected one of {allowed}")
    return f"constrained output {text!r}"


def check_embeddings(client: FireworksClient) -> str:
    embedder = FireworksEmbedder(client)
    vectors = embedder.embed_documents([
        "Low recall on the minority open-circuit class due to class imbalance.",
        "Weighted sampling improved minority recall on an imbalanced PCB dataset.",
        "The learning rate was too high and training diverged.",
    ])
    dims = len(vectors[0])
    related = cosine_similarity(vectors[0], vectors[1])
    unrelated = cosine_similarity(vectors[0], vectors[2])
    if related <= unrelated:
        raise RuntimeError(
            f"similarity is not discriminating: related={related:.3f} "
            f"<= unrelated={unrelated:.3f}"
        )
    return (f"{dims}-dim vectors, related={related:.3f} > unrelated={unrelated:.3f}; "
            f"Atlas numDimensions must be {dims}")


def check_rerank(client: FireworksClient) -> str:
    documents = [
        "Raising input resolution helped tiny defects become detectable.",
        "Weighted sampling fixed low recall on a rare defect class.",
        "Switching the optimizer made no measurable difference.",
    ]
    rows = client.rerank(
        "Minority defect class has very low recall because of class imbalance.",
        documents, top_n=3, purpose="smoke.rerank",
    )
    if not rows:
        raise RuntimeError("rerank returned no rows")
    best = int(rows[0]["index"])
    note = "" if best == 1 else f" {YELLOW}(expected doc 1 to rank first){RESET}"
    return (f"top={best} score={rows[0]['relevance_score']:.3f}, "
            f"{len(rows)} scored{note}")


def check_vision(client: FireworksClient) -> str:
    from pcb_agent.fireworks.vision import build_image_message

    # A 2x2 PNG, generated inline so the check needs no fixture files.
    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAF0lEQVR4nGP8//8/A"
        "xJgYkAFo3wG6vEBAJhcAwFbxUCoAAAAAElFTkSuQmCC"
    )
    message = build_image_message("How many distinct colours are in this image?",
                                  [tiny_png])
    payload = client.chat([message], role=Role.VISION, max_tokens=64,
                          temperature=0.0, purpose="smoke.vision")
    text = (payload["choices"][0]["message"]["content"] or "").strip()
    return f"VLM responded {text[:60]!r}"


CHECKS: dict[str, tuple[str, Callable[[FireworksClient], str]]] = {
    "models": ("Model registry vs live catalogue", check_models),
    "chat": ("Chat completions", check_chat),
    "streaming": ("Streaming (SSE)", check_streaming),
    "structured": ("Structured output (json_schema)", check_structured),
    "tools": ("Tool calling (experiment allowlist)", check_tools),
    "grammar": ("Grammar mode (BNF)", check_grammar),
    "embeddings": ("Embeddings for Atlas Vector Search", check_embeddings),
    "rerank": ("Reranking", check_rerank),
    "vision": ("Vision (PCB crop inspection)", check_vision),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated subset: " + ",".join(CHECKS))
    parser.add_argument("--models", action="store_true",
                        help="shorthand for --only models")
    parser.add_argument("--verbose", action="store_true", help="print tracebacks")
    args = parser.parse_args(argv)

    _load_dotenv()

    selected = list(CHECKS)
    if args.models:
        selected = ["models"]
    elif args.only:
        selected = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = set(selected) - set(CHECKS)
        if unknown:
            print(f"unknown check(s): {sorted(unknown)}", file=sys.stderr)
            return 2

    try:
        settings = FireworksSettings.from_env()
    except FireworksError as exc:
        print(f"{RED}config error:{RESET} {exc}", file=sys.stderr)
        return 2

    print(f"base_url: {settings.base_url}")
    print(f"key:      {settings.redacted()['api_key']}")
    print()

    client = FireworksClient(settings, ledger=UsageLedger(run_id="smoke"))
    failures: list[str] = []

    try:
        for name in selected:
            label, fn = CHECKS[name]
            print(f"  {label:<40}", end="", flush=True)
            started = time.time()
            try:
                detail = fn(client)
            except Exception as exc:  # noqa: BLE001 - a smoke test reports, never raises
                elapsed = time.time() - started
                print(f"{RED}FAIL{RESET} ({elapsed:.1f}s)")
                print(f"      {RED}{type(exc).__name__}: {exc}{RESET}")
                if args.verbose:
                    traceback.print_exc()
                failures.append(name)
            else:
                elapsed = time.time() - started
                print(f"{GREEN}ok{RESET}   ({elapsed:.1f}s)")
                print(f"      {DIM}{detail}{RESET}")
    finally:
        client.close()

    print()
    print(client.ledger.summary_line())
    by_purpose = client.ledger.by_purpose()
    if by_purpose:
        print(DIM + json.dumps(by_purpose, indent=2) + RESET)

    if failures:
        print(f"\n{RED}{len(failures)} of {len(selected)} checks failed: "
              f"{', '.join(failures)}{RESET}")
        print("See FIREWORKS.md > Troubleshooting.")
        return 1

    print(f"\n{GREEN}all {len(selected)} checks passed{RESET}")
    return 0


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
