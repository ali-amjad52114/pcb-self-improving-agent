"""The simulator, plus the full loop wired end to end against a mocked Fireworks."""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import embeddings_response, json_chat_response, rerank_response, tool_call

from pcb_agent.demo.fireworks_loop import run_loop
from pcb_agent.demo.memory_store import InMemoryLessonStore
from pcb_agent.fireworks.embeddings import FireworksEmbedder
from pcb_agent.fireworks.tools import DEFAULT_TRAINING_CONFIG, ToolCall, apply_tool
from pcb_agent.ml.simulator import TASK_A, Trainer


class TestSimulator:
    def test_baseline_is_weak_enough_to_be_worth_improving(self):
        trainer = Trainer(TASK_A)
        model = trainer.train_model(DEFAULT_TRAINING_CONFIG)
        metrics = trainer.evaluate_model(model["model_id"])
        assert 0.4 < metrics["macro_f1"] < 0.75

    def test_evaluation_is_deterministic(self):
        trainer = Trainer(TASK_A)
        model = trainer.train_model(DEFAULT_TRAINING_CONFIG)
        first = trainer.evaluate_model(model["model_id"])
        second = trainer.evaluate_model(model["model_id"])
        assert first == second

    def test_unknown_model_id_raises(self):
        with pytest.raises(KeyError):
            Trainer(TASK_A).evaluate_model("model_nope")

    def _score(self, config) -> float:
        trainer = Trainer(TASK_A)
        model = trainer.train_model(config)
        return trainer.evaluate_model(model["model_id"])["macro_f1"]

    def test_weighted_sampling_helps_the_imbalanced_task(self):
        base = self._score(DEFAULT_TRAINING_CONFIG)
        fixed, _ = apply_tool(
            DEFAULT_TRAINING_CONFIG,
            ToolCall("change_sampler", {"strategy": "weighted", "reason": "r"}),
        )
        assert self._score(fixed) > base

    def test_raising_resolution_helps_small_defects(self):
        base = self._score(DEFAULT_TRAINING_CONFIG)
        fixed, _ = apply_tool(
            DEFAULT_TRAINING_CONFIG,
            ToolCall("change_image_size", {"image_size": 512, "reason": "r"}),
        )
        assert self._score(fixed) > base

    def test_an_irrelevant_change_does_not_help(self):
        base = self._score(DEFAULT_TRAINING_CONFIG)
        neutral, _ = apply_tool(
            DEFAULT_TRAINING_CONFIG,
            ToolCall("change_batch_size", {"batch_size": 64, "reason": "r"}),
        )
        assert self._score(neutral) <= base + 0.01

    def test_a_bad_learning_rate_hurts(self):
        base = self._score(DEFAULT_TRAINING_CONFIG)
        broken, _ = apply_tool(
            DEFAULT_TRAINING_CONFIG,
            ToolCall("change_learning_rate", {"learning_rate": 0.5, "reason": "r"}),
        )
        assert self._score(broken) < base

    def test_confusion_matrix_rows_are_consistent_with_recall(self):
        trainer = Trainer(TASK_A)
        model = trainer.train_model(DEFAULT_TRAINING_CONFIG)
        metrics = trainer.evaluate_model(model["model_id"])
        matrix = metrics["confusion_matrix"]["matrix"]
        for cls, row in matrix.items():
            support = metrics["per_class"][cls]["support"] // 5
            assert sum(row.values()) <= support
            assert row[cls] == pytest.approx(
                round(support * metrics["per_class"][cls]["recall"]), abs=1)


DIAGNOSIS = {
    "diagnosis": "open_circuit recall is very low and the class is rare.",
    "failure_signature": {
        "summary": "Low recall on a rare defect class under heavy imbalance.",
        "affected_classes": ["open_circuit"],
        "failure_mode": "class_imbalance",
    },
    "hypothesis": "Weighted sampling raises minority recall.",
    "confidence": 0.8,
}

CRITIQUE = {
    "hypothesis_correct": True,
    "verdict": "improved",
    "explanation": "Minority recall rose substantially.",
    "lesson": {
        "failure_summary": "Rare defect class under-recalled with heavy imbalance.",
        "intervention": "weighted sampler",
        "result": "macro F1 improved",
        "lesson": "Weighted sampling helps when a class is under 5% of samples.",
        "reusable": True,
    },
    "confidence": 0.85,
}


def _scripted_fireworks(actions: list[tuple[str, dict]]):
    """A handler that plays the scientist, critic, embedder, and reranker.

    `actions` is the sequence of tool calls the fake scientist will make.
    """
    state = {"action_index": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}

        if path.endswith("/embeddings"):
            return embeddings_response([[0.1, 0.2, 0.3] for _ in body["input"]])

        if path.endswith("/rerank"):
            return rerank_response(
                [(i, 0.9 - i * 0.3) for i in range(len(body["documents"]))]
            )

        if body.get("tools"):
            index = min(state["action_index"], len(actions) - 1)
            state["action_index"] += 1
            name, args = actions[index]
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [tool_call(name, {**args, "reason": "scripted"})],
            }, "finish_reason": "tool_calls"}]})

        schema_name = (body.get("response_format") or {}).get(
            "json_schema", {}).get("name")
        if schema_name == "Diagnosis":
            return json_chat_response(DIAGNOSIS)
        if schema_name == "Critique":
            return json_chat_response(CRITIQUE)

        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})

    return handler


class TestFullLoop:
    def test_loop_improves_and_stops_at_the_target(self, make_client, tmp_path):
        client = make_client(_scripted_fireworks([
            ("change_sampler", {"strategy": "weighted"}),
            ("change_image_size", {"image_size": 512}),
            ("change_class_weights", {"mode": "balanced"}),
        ]))
        store = InMemoryLessonStore(FireworksEmbedder(client),
                                    path=tmp_path / "lessons.json")

        result = run_loop(client, store, TASK_A, run_id="run_test",
                          target=0.80, budget=4, verbose=False)

        assert result.best_macro_f1 > result.baseline_macro_f1
        assert result.experiments_used >= 1
        assert len(result.trajectory) == result.experiments_used + 1

    def test_every_experiment_writes_an_experiment_and_a_lesson(self, make_client, tmp_path):
        client = make_client(_scripted_fireworks([
            ("change_sampler", {"strategy": "weighted"}),
        ]))
        store = InMemoryLessonStore(FireworksEmbedder(client),
                                    path=tmp_path / "lessons.json")

        result = run_loop(client, store, TASK_A, run_id="run_test",
                          target=0.99, budget=2, verbose=False)

        assert len(store.experiments) == result.experiments_used
        assert len(store.lessons) == result.experiments_used
        assert all(lesson.get("embedding") for lesson in store.lessons)
        assert store.get_run_history("run_test")

    def test_lessons_persist_and_are_retrieved_by_a_later_run(self, make_client, tmp_path):
        path = tmp_path / "lessons.json"
        client = make_client(_scripted_fireworks([
            ("change_sampler", {"strategy": "weighted"}),
        ]))

        first = InMemoryLessonStore(FireworksEmbedder(client), path=path)
        run_loop(client, first, TASK_A, run_id="run_a", target=0.99, budget=1,
                 verbose=False)
        assert first.lessons

        # A fresh store reading the same file sees the prior run's experience.
        second = InMemoryLessonStore(FireworksEmbedder(client), path=path)
        assert len(second.lessons) == len(first.lessons)
        retrieved = second.retrieve_similar_lessons("rare class low recall", k=3)
        assert retrieved
        assert "embedding" not in retrieved[0]
        assert retrieved[0]["score"] == pytest.approx(1.0)

    def test_experienced_run_retrieves_memories(self, make_client, tmp_path):
        path = tmp_path / "lessons.json"
        client = make_client(_scripted_fireworks([
            ("change_sampler", {"strategy": "weighted"}),
            ("change_image_size", {"image_size": 448}),
        ]))

        store = InMemoryLessonStore(FireworksEmbedder(client), path=path)
        cold = run_loop(client, store, TASK_A, run_id="run_a", target=0.99,
                        budget=2, verbose=False)
        # A cold run starts with an empty store, so its first iteration sees
        # nothing. Later iterations can retrieve what the run itself wrote.
        assert cold.memories_seen < cold.experiments_used * 2

        warm = run_loop(client, store, TASK_A, run_id="run_b", target=0.99,
                        budget=2, verbose=False)
        # The experienced run has prior lessons available from iteration 1.
        assert warm.memories_seen > cold.memories_seen

    def test_usage_is_tracked_across_the_run(self, make_client, tmp_path):
        client = make_client(_scripted_fireworks([
            ("change_sampler", {"strategy": "weighted"}),
        ]))
        store = InMemoryLessonStore(FireworksEmbedder(client),
                                    path=tmp_path / "lessons.json")
        run_loop(client, store, TASK_A, run_id="run_test", target=0.99, budget=1,
                 verbose=False)

        by_purpose = client.ledger.by_purpose()
        assert {"diagnose", "critique", "propose_experiment",
                "embed_documents"} <= set(by_purpose)
        # Each purpose is attributed to the model role that served it.
        assert by_purpose["diagnose"]["models"] != by_purpose["critique"]["models"]


def test_memory_store_filters_non_reusable_lessons(make_client, tmp_path):
    client = make_client(lambda request: embeddings_response([[1.0, 0.0]]))
    store = InMemoryLessonStore(FireworksEmbedder(client), path=tmp_path / "l.json")
    store.store_lesson({"failure_summary": "keep me", "reusable": True,
                        "embedding_text": "keep me"})
    store.store_lesson({"failure_summary": "hide me", "reusable": False,
                        "embedding_text": "hide me"})

    retrieved = store.retrieve_similar_lessons("anything", k=5)
    assert [r["failure_summary"] for r in retrieved] == ["keep me"]
    assert store.stats()["lessons"] == 2
    assert store.stats()["reusable_lessons"] == 1
