"""Small HTTP API for running and observing PCB agent campaigns.

The module intentionally uses the standard library HTTP server so the API adds no
runtime dependency to the existing backend.  Campaigns run in a bounded worker
pool and LangGraph node updates are copied into an in-memory, JSON-safe snapshot.
Durable campaign state remains owned by the configured LangGraph checkpointer.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import re
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from pcb_agent.config import Settings
from pcb_agent.runner import (
    _build_runtime,
    _initial_state,
    _load_baseline_config,
)


MAX_BODY_BYTES = 1_000_000
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
DEFAULT_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://traceboard-pcb-lab.aamjad52114.chatgpt.site",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Recursively convert common scientific/database values to JSON types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return json_safe(model_dump(mode="json"))
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass
    # ObjectId and other checkpoint-friendly scalar types have stable strings.
    return str(value)


def _public_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"[:500]
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", message)
    message = re.sub(r"fw_[A-Za-z0-9_-]+", "[redacted]", message)
    message = re.sub(r"(mongodb(?:\+srv)?://)[^@\s]+@", r"\1[redacted]@", message)
    return message


@dataclass
class RunRecord:
    run_id: str
    mode: str
    status: str = "queued"
    current_node: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def snapshot(self, *, include_state: bool = True) -> dict[str, Any]:
        result = asdict(self)
        # Compatibility alias used by the dashboard's stage visualizer.
        result["current_stage"] = self.current_node
        if not include_state:
            result.pop("state", None)
        return json_safe(result)


RuntimeFactory = Callable[[], tuple[Settings, Any, Any, Any]]


class RunManager:
    """Thread-safe registry plus a bounded pool for expensive ML campaigns."""

    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory = _build_runtime,
        max_workers: int = 1,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="pcb-campaign"
        )
        self._lock = threading.RLock()
        self._records: dict[str, RunRecord] = {}
        self._futures: dict[str, Future[None]] = {}

    def start(
        self,
        *,
        mode: str,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode = mode.strip().lower()
        if mode not in {"cold", "memory"}:
            raise ValueError("mode must be 'cold' or 'memory'")
        run_id = run_id or f"web_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run_id must be 1-64 letters, numbers, '.', '_' or '-'")

        baseline = _load_baseline_config(None)
        overrides = dict(config or {})
        unknown = sorted(set(overrides) - set(baseline))
        if unknown:
            raise ValueError(f"unknown config fields: {', '.join(unknown)}")
        baseline.update(overrides)

        with self._lock:
            if run_id in self._records:
                raise KeyError(run_id)
            record = RunRecord(run_id=run_id, mode=mode)
            self._records[run_id] = record
            self._futures[run_id] = self._executor.submit(
                self._execute, run_id, mode, baseline
            )
            return record.snapshot()

    def _update(self, run_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._records[run_id]
            for key, value in changes.items():
                setattr(record, key, json_safe(value) if key == "state" else value)

    def _execute(self, run_id: str, mode: str, baseline: dict[str, Any]) -> None:
        self._update(
            run_id,
            status="running",
            current_node="initializing",
            started_at=_utc_now(),
        )
        try:
            settings, deps, _checkpointer, graph = self._runtime_factory()
            summary_getter = getattr(deps.ml, "get_dataset_summary", None)
            dataset_summary = (
                dict(summary_getter(baseline)) if callable(summary_getter) else {}
            )
            state = dict(
                _initial_state(
                    run_id=run_id,
                    settings=settings,
                    config=baseline,
                    dataset_summary=dataset_summary,
                    memory_mode=mode,
                )
            )
            self._update(run_id, state=state)
            thread_config = {
                "configurable": {"thread_id": run_id},
                "run_name": "pcb-agent-api-campaign",
                "tags": ["pcb-agent", "api", f"mode:{mode}"],
                "metadata": {"pcb_run_id": run_id, "memory_mode": mode},
            }

            for event in graph.stream(state, config=thread_config, stream_mode="updates"):
                for node, update in dict(event or {}).items():
                    if isinstance(update, Mapping):
                        state.update(update)
                    self._update(
                        run_id,
                        status="running",
                        current_node=str(node),
                        state=state,
                    )

            snapshot = graph.get_state(thread_config)
            if snapshot and snapshot.values:
                state.update(dict(snapshot.values))

            test_evaluator = getattr(deps.ml, "evaluate_test", None)
            if callable(test_evaluator) and settings.agent_mode == "integrated":
                try:
                    test_result = dict(
                        test_evaluator(dict(state.get("best_config") or baseline))
                    )
                    state["final_test_metrics"] = dict(
                        test_result.get("metrics") or test_result
                    )
                except Exception as exc:  # final test must not discard the campaign
                    warnings = list(state.get("warnings") or [])
                    warnings.append(f"final_test_evaluation_failed: {exc}")
                    state["warnings"] = warnings

            self._update(
                run_id,
                status="complete",
                current_node="complete",
                completed_at=_utc_now(),
                state=state,
            )
        except BaseException as exc:
            self._update(
                run_id,
                status="failed",
                current_node="failed",
                completed_at=_utc_now(),
                error=_public_error(exc),
            )

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return self._records[run_id].snapshot()
            except KeyError:
                raise KeyError(run_id) from None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(
                self._records.values(), key=lambda item: item.created_at, reverse=True
            )
            return [record.snapshot(include_state=False) for record in records]

    def history(self, run_id: str) -> list[dict[str, Any]]:
        record = self.get(run_id)
        return list(record.get("state", {}).get("experiment_history") or [])

    def wait(self, run_id: str, timeout: float = 10.0) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(run_id)
        if future is None:
            raise KeyError(run_id)
        future.result(timeout=timeout)
        return self.get(run_id)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)


def _origins_from_env() -> tuple[str, ...]:
    raw = os.getenv("PCB_API_CORS_ORIGINS", "")
    return tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip()) or DEFAULT_ORIGINS


class APIHandler(BaseHTTPRequestHandler):
    manager: RunManager
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS
    api_token: str = ""
    server_version = "PCB-Agent-API/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep standard access logging but avoid reverse-DNS or request bodies.
        super().log_message(format, *args)

    def _cors_origin(self) -> str | None:
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if "*" in self.allowed_origins:
            return "*"
        return origin if origin in self.allowed_origins else None

    def _send(self, status: int, payload: Any) -> None:
        encoded = json.dumps(json_safe(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(encoded)

    def _authorized(self) -> bool:
        if not self.api_token:
            return True
        authorization = self.headers.get("Authorization") or ""
        scheme, _, credential = authorization.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(
            credential, self.api_token
        )

    def _require_authorization(self) -> bool:
        if self._authorized():
            return True
        self._send(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body must be non-empty and at most 1 MB")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._require_authorization():
            return
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in {"/health", "/api/health"}:
            self._send(HTTPStatus.OK, {"status": "ok", "service": "pcb-agent-api"})
            return
        if path in {"/runs", "/api/runs"}:
            self._send(HTTPStatus.OK, {"runs": self.manager.list()})
            return
        match = re.fullmatch(r"/(?:api/)?runs/([^/]+)(?:/(state|history))?", path)
        if match:
            run_id, section = match.groups()
            try:
                record = self.manager.get(run_id)
                if section == "state":
                    payload: Any = record["state"]
                elif section == "history":
                    payload = {"run_id": run_id, "history": self.manager.history(run_id)}
                else:
                    payload = record
                self._send(HTTPStatus.OK, payload)
            except KeyError:
                self._send(HTTPStatus.NOT_FOUND, {"error": "run not found"})
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_authorization():
            return
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path not in {"/runs", "/api/runs"}:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            body = self._body()
            unknown = sorted(set(body) - {"run_id", "mode", "config"})
            if unknown:
                raise ValueError(f"unknown request fields: {', '.join(unknown)}")
            config = body.get("config")
            if config is not None and not isinstance(config, dict):
                raise ValueError("config must be a JSON object")
            record = self.manager.start(
                mode=str(body.get("mode", "memory")),
                run_id=body.get("run_id"),
                config=config,
            )
            self._send(HTTPStatus.ACCEPTED, record)
        except KeyError:
            self._send(HTTPStatus.CONFLICT, {"error": "run_id already exists"})
        except ValueError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def make_server(
    host: str,
    port: int,
    *,
    manager: RunManager | None = None,
    allowed_origins: tuple[str, ...] | None = None,
    api_token: str | None = None,
) -> ThreadingHTTPServer:
    manager = manager or RunManager(
        max_workers=int(os.getenv("PCB_API_MAX_WORKERS", "1"))
    )

    class BoundHandler(APIHandler):
        pass

    BoundHandler.manager = manager
    BoundHandler.allowed_origins = allowed_origins or _origins_from_env()
    BoundHandler.api_token = (
        os.getenv("PCB_API_TOKEN", "") if api_token is None else api_token
    )
    server = ThreadingHTTPServer((host, port), BoundHandler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="PCB agent dashboard API")
    parser.add_argument("--host", default=os.getenv("PCB_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PCB_API_PORT", "8000")))
    args = parser.parse_args()
    manager = RunManager(max_workers=int(os.getenv("PCB_API_MAX_WORKERS", "1")))
    server = make_server(args.host, args.port, manager=manager)
    print(f"PCB agent API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        manager.shutdown(wait=True)


if __name__ == "__main__":
    main()
