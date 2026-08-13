"""Focused API tests with no network services or external model calls."""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from pcb_agent.api import RunManager, json_safe, make_server
from pcb_agent.config import Settings
from pcb_agent.dependencies import AgentDependencies
from pcb_agent.fakes.judge import FakeJudgeAdapter
from pcb_agent.fakes.memory import FakeMemory
from pcb_agent.fakes.ml import FakeML
from pcb_agent.fakes.reasoning import FakeReasoning
from pcb_agent.graph import build_graph


def _runtime():
    from langgraph.checkpoint.memory import InMemorySaver

    settings = Settings(
        _env_file=None,
        agent_mode="fake",
        target_metric="macro_f1",
        target_value=0.6,
        experiment_budget=1,
        checkpointer_backend="memory",
        openrouter_enabled=False,
    )
    deps = AgentDependencies(
        memory=FakeMemory(),
        ml=FakeML(),
        reasoning=FakeReasoning(),
        judge=FakeJudgeAdapter(),
    )
    saver = InMemorySaver()
    return settings, deps, saver, build_graph(deps, saver, settings)


@pytest.fixture
def manager():
    value = RunManager(runtime_factory=_runtime, max_workers=1)
    yield value
    value.shutdown()


def test_run_manager_exposes_real_graph_state_and_history(manager: RunManager):
    started = manager.start(mode="cold", run_id="api-test")
    assert started["status"] in {"queued", "running"}

    completed = manager.wait("api-test")
    assert completed["status"] == "complete"
    assert completed["current_node"] == "complete"
    assert completed["state"]["run_id"] == "api-test"
    assert completed["state"]["baseline_metrics"]["macro_f1"] == 0.55
    assert manager.history("api-test") == completed["state"]["experiment_history"]
    assert manager.list()[0]["run_id"] == "api-test"


def test_run_manager_rejects_bad_input_and_duplicate_ids(manager: RunManager):
    with pytest.raises(ValueError, match="mode"):
        manager.start(mode="unknown")
    with pytest.raises(ValueError, match="run_id"):
        manager.start(mode="cold", run_id="bad id")
    with pytest.raises(ValueError, match="unknown config"):
        manager.start(mode="cold", config={"shell_command": "no"})

    manager.start(mode="memory", run_id="unique")
    with pytest.raises(KeyError):
        manager.start(mode="memory", run_id="unique")


def test_json_safe_handles_nonfinite_and_scientific_scalars():
    class Scalar:
        def item(self):
            return 7

    assert json_safe({"nan": float("nan"), "scalar": Scalar()}) == {
        "nan": None,
        "scalar": 7,
    }


def test_http_health_start_status_and_cors(manager: RunManager):
    server = make_server(
        "127.0.0.1",
        0,
        manager=manager,
        allowed_origins=("https://dashboard.example",),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(f"{base}/health") as response:
            assert json.load(response)["status"] == "ok"

        request = Request(
            f"{base}/runs",
            data=json.dumps({"run_id": "http-test", "mode": "memory"}).encode(),
            headers={
                "Content-Type": "application/json",
                "Origin": "https://dashboard.example",
            },
            method="POST",
        )
        with urlopen(request) as response:
            assert response.status == 202
            assert response.headers["Access-Control-Allow-Origin"] == "https://dashboard.example"

        manager.wait("http-test")
        with urlopen(f"{base}/runs/http-test") as response:
            assert json.load(response)["status"] == "complete"
        with urlopen(f"{base}/api/runs/http-test/history") as response:
            assert json.load(response)["history"]

        with pytest.raises(HTTPError) as missing:
            urlopen(f"{base}/api/runs/missing")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_bearer_token_protects_all_get_and_post_routes(manager: RunManager):
    server = make_server("127.0.0.1", 0, manager=manager, api_token="test-secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(f"{base}/health")
        assert unauthorized.value.code == 401

        request = Request(
            f"{base}/health", headers={"Authorization": "Bearer test-secret"}
        )
        with urlopen(request) as response:
            assert json.load(response)["status"] == "ok"

        post = Request(
            f"{base}/api/runs",
            data=b'{"mode":"cold"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as unauthorized_post:
            urlopen(post)
        assert unauthorized_post.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
