"""Scientist, critic, memory retrieval, and embeddings, all against mocks."""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import chat_response, embeddings_response, json_chat_response, rerank_response, tool_call

from pcb_agent.fireworks.critic import (
    _delta,
    _verdict,
    critique_result,
    embedding_text,
    to_lesson_document,
)
from pcb_agent.fireworks.embeddings import FireworksEmbedder, attach_embeddings, cosine_similarity
from pcb_agent.fireworks.errors import FireworksToolChoiceError
from pcb_agent.fireworks.rerank import (
    actions_to_withdraw,
    rerank_lessons,
    retrieve_memories,
    split_by_polarity,
)
from pcb_agent.fireworks.scientist import diagnose, propose_experiment, validate_proposal

STATE = {
    "run_id": "run_a",
    "dataset_summary": {"n_samples": 3680, "imbalance_ratio": 12.6},
    "current_config": {"sampler": "random", "image_size": 224},
    "current_metrics": {"macro_f1": 0.58},
    "confusion_matrix": {"labels": ["open_circuit", "short"]},
    "experiment_history": [],
}

DIAGNOSIS = {
    "diagnosis": "open_circuit is under-recalled because it is 4% of the data.",
    "failure_signature": {
        "summary": "Low recall on a rare defect class under heavy imbalance.",
        "affected_classes": ["open_circuit"],
        "failure_mode": "class_imbalance",
    },
    "hypothesis": "Weighted sampling raises minority recall.",
    "confidence": 0.78,
}


class TestDiagnose:
    def test_returns_the_validated_diagnosis(self, make_client):
        client = make_client([json_chat_response(DIAGNOSIS)])
        result = diagnose(client, STATE)
        assert result["failure_signature"]["failure_mode"] == "class_imbalance"
        assert result["cold_start"] is True

    def test_cold_start_flag_is_false_when_memories_exist(self, make_client):
        client = make_client([json_chat_response(DIAGNOSIS)])
        result = diagnose(client, STATE, [{"lesson_id": "lesson_1",
                                           "failure_summary": "x"}])
        assert result["cold_start"] is False

    def test_hallucinated_lesson_citations_are_dropped(self, make_client):
        payload = dict(DIAGNOSIS, memory_used=["lesson_1", "lesson_999"])
        client = make_client([json_chat_response(payload)])
        result = diagnose(client, STATE, [{"lesson_id": "lesson_1"}])
        assert result["memory_used"] == ["lesson_1"]

    def test_cold_start_prompt_says_so(self, make_client):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return json_chat_response(DIAGNOSIS)

        client = make_client(handler)
        diagnose(client, STATE)
        prompt = captured["messages"][-1]["content"]
        assert "cold start" in prompt

    def test_retrieved_lessons_are_rendered_with_polarity(self, make_client):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return json_chat_response(DIAGNOSIS)

        client = make_client(handler)
        diagnose(client, STATE, [
            {"lesson_id": "l1", "failure_summary": "imbalance", "avoid": False},
            {"lesson_id": "l2", "failure_summary": "bad idea", "avoid": True},
        ])
        prompt = captured["messages"][-1]["content"]
        assert "[USE] l1" in prompt
        assert "[AVOID] l2" in prompt


class TestProposeExperiment:
    def _payload(self, name="change_sampler", args=None):
        args = args or {"strategy": "weighted", "reason": "minority recall"}
        return {"choices": [{"message": {"role": "assistant", "content": None,
                                         "tool_calls": [tool_call(name, args)]}}]}

    def test_returns_an_executable_proposal(self, make_client):
        client = make_client([httpx.Response(200, json=self._payload())])
        proposal = propose_experiment(client, STATE, DIAGNOSIS)
        assert proposal["next_action"] == "change_sampler"
        assert proposal["parameters"] == {"strategy": "weighted"}
        assert proposal["rationale"] == "minority recall"

    def test_tool_choice_is_required_and_parallel_is_off(self, make_client):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=self._payload())

        client = make_client(handler)
        propose_experiment(client, STATE, DIAGNOSIS)
        assert captured["tool_choice"] == "required"
        assert captured["parallel_tool_calls"] is False
        assert len(captured["tools"]) == 9

    def test_narrowed_allowlist_is_sent_and_explained(self, make_client):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=self._payload())

        client = make_client(handler)
        propose_experiment(client, STATE, DIAGNOSIS,
                           allowed_actions=["change_sampler", "change_image_size",
                                            "change_class_weights"])
        names = [t["function"]["name"] for t in captured["tools"]]
        assert names == ["change_sampler", "change_image_size", "change_class_weights"]
        assert "WITHDRAWN ACTIONS" in captured["messages"][-1]["content"]

    def test_a_withdrawn_action_is_rejected_if_the_model_calls_it_anyway(self, make_client):
        client = make_client([httpx.Response(200, json=self._payload("retrain",
                                                                    {"reason": "r"}))])
        with pytest.raises(FireworksToolChoiceError, match="excluded"):
            propose_experiment(client, STATE, DIAGNOSIS,
                               allowed_actions=["change_sampler", "change_image_size",
                                                "change_class_weights"])

    def test_low_confidence_requests_a_second_opinion(self, make_client):
        client = make_client([httpx.Response(200, json=self._payload())])
        proposal = propose_experiment(client, STATE, dict(DIAGNOSIS, confidence=0.3))
        assert proposal["second_opinion_requested"] is True

    def test_two_consecutive_failures_request_a_second_opinion(self, make_client):
        client = make_client([httpx.Response(200, json=self._payload())])
        state = dict(STATE, experiment_history=[
            {"action": "a", "delta": -0.02}, {"action": "b", "delta": 0.0}])
        proposal = propose_experiment(client, state, DIAGNOSIS)
        assert proposal["second_opinion_requested"] is True

    def test_force_pins_a_specific_tool(self, make_client):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=self._payload())

        client = make_client(handler)
        propose_experiment(client, STATE, DIAGNOSIS, force="change_sampler")
        assert captured["tool_choice"] == {"type": "function",
                                           "function": {"name": "change_sampler"}}


def test_validate_proposal_catches_bad_parameters():
    errors = validate_proposal({
        "next_action": "change_image_size",
        "parameters": {"image_size": 99999},
        "expected_effect": "x", "confidence": 0.5,
    })
    assert errors


def test_validate_proposal_accepts_a_good_one():
    assert validate_proposal({
        "next_action": "change_sampler",
        "parameters": {"strategy": "weighted", "reason": "r"},
        "expected_effect": "raise macro F1", "confidence": 0.7,
    }) == []


CRITIQUE = {
    "hypothesis_correct": True,
    "verdict": "improved",
    "explanation": "Minority recall rose from 0.32 to 0.61.",
    "lesson": {
        "failure_summary": "Rare defect class under-recalled with heavy imbalance.",
        "intervention": "weighted sampler",
        "result": "+0.11 macro F1",
        "lesson": "Weighted sampling helps when a class is under 5% of samples.",
        "reusable": True,
    },
    "confidence": 0.86,
}


class TestCritic:
    def test_delta_is_computed_for_scalars_and_per_class(self):
        before = {"macro_f1": 0.58, "per_class": {"a": {"recall": 0.3}}}
        after = {"macro_f1": 0.69, "per_class": {"a": {"recall": 0.61}}}
        delta = _delta(before, after)
        assert delta["macro_f1"] == pytest.approx(0.11)
        assert delta["per_class.a.recall"] == pytest.approx(0.31)

    @pytest.mark.parametrize("value,expected", [
        (0.05, "improved"), (-0.05, "regressed"), (0.002, "no_change"),
    ])
    def test_verdict_thresholds(self, value, expected):
        assert _verdict({"macro_f1": value}, 0.01) == expected

    def test_measured_verdict_overrides_the_model(self, make_client):
        client = make_client([json_chat_response(CRITIQUE)])
        result = critique_result(
            client, {"macro_f1": 0.58}, {"macro_f1": 0.575},
            {"next_action": "change_sampler"},
        )
        # The model claimed "improved"; the numbers say otherwise.
        assert result["verdict"] == "no_change"
        assert result["lesson"]["reusable"] is False

    def test_improvement_is_kept_when_the_numbers_agree(self, make_client):
        client = make_client([json_chat_response(CRITIQUE)])
        result = critique_result(
            client, {"macro_f1": 0.58}, {"macro_f1": 0.69},
            {"next_action": "change_sampler"},
        )
        assert result["verdict"] == "improved"
        assert result["lesson"]["avoid"] is False

    def test_regression_marks_the_lesson_as_avoid(self, make_client):
        client = make_client([json_chat_response(CRITIQUE)])
        result = critique_result(
            client, {"macro_f1": 0.69}, {"macro_f1": 0.55},
            {"next_action": "change_model"},
        )
        assert result["verdict"] == "regressed"
        assert result["lesson"]["avoid"] is True

    def test_calibration_scores_the_scientists_forecast(self, make_client):
        client = make_client([json_chat_response(CRITIQUE)])
        result = critique_result(
            client, {"macro_f1": 0.58}, {"macro_f1": 0.69},
            {"next_action": "change_sampler", "expected_delta_macro_f1": 0.08},
        )
        assert result["calibration"]["error"] == pytest.approx(0.03)
        assert result["calibration"]["direction_correct"] is True

    def test_to_lesson_document_is_mongodb_ready(self, make_client):
        client = make_client([json_chat_response(CRITIQUE)])
        critique = critique_result(client, {"macro_f1": 0.58}, {"macro_f1": 0.69},
                                   {"next_action": "change_sampler"})
        doc = to_lesson_document(
            critique, experiment_id="exp_001", run_id="run_a",
            failure_signature=DIAGNOSIS["failure_signature"],
        )
        assert doc["run_id"] == "run_a"
        assert doc["failure_mode"] == "class_imbalance"
        assert doc["affected_classes"] == ["open_circuit"]
        assert doc["embedding_text"]
        assert doc["delta_macro_f1"] == pytest.approx(0.11)


def test_embedding_text_describes_the_problem_not_the_fix():
    text = embedding_text(
        {"failure_summary": "Rare class under-recalled.",
         "lesson": "Weighted sampling helps under 5% prevalence.",
         "intervention": "weighted sampler"},
        {"failure_mode": "class_imbalance", "affected_classes": ["open_circuit"]},
    )
    assert "Rare class under-recalled." in text
    assert "class_imbalance" in text
    assert "open_circuit" in text


class TestRetrieval:
    CANDIDATES = [
        {"lesson_id": "l1", "embedding_text": "imbalance hurt minority recall",
         "score": 0.81},
        {"lesson_id": "l2", "embedding_text": "learning rate was too high", "score": 0.79},
        {"lesson_id": "l3", "embedding_text": "unrelated note", "score": 0.77},
    ]

    def test_rerank_reorders_and_applies_the_relevance_floor(self, make_client):
        client = make_client([rerank_response([(0, 0.91), (1, 0.20), (2, 0.05)])])
        ranked = rerank_lessons(client, "minority recall problem", self.CANDIDATES)
        assert [r.lesson["lesson_id"] for r in ranked] == ["l1"]
        assert ranked[0].vector_score == pytest.approx(0.81)
        assert ranked[0].rerank_score == pytest.approx(0.91)

    def test_everything_below_the_floor_yields_an_honest_cold_start(self, make_client):
        client = make_client([rerank_response([(0, 0.1), (1, 0.05), (2, 0.01)])])
        assert rerank_lessons(client, "unrelated", self.CANDIDATES) == []

    def test_no_candidates_makes_no_request(self, make_client):
        client = make_client(lambda request: pytest.fail("should not be called"))
        assert retrieve_memories(client, "q", []) == []

    def test_rerank_failure_falls_back_to_vector_order(self, make_client):
        client = make_client(lambda request: httpx.Response(500, json={"error": "x"}))
        memories = retrieve_memories(client, "q", self.CANDIDATES, top_k=2)
        assert [m["lesson_id"] for m in memories] == ["l1", "l2"]
        assert memories[0]["rerank_score"] is None

    def test_split_by_polarity(self):
        use, avoid = split_by_polarity([{"lesson_id": "a"},
                                        {"lesson_id": "b", "avoid": True}])
        assert [m["lesson_id"] for m in use] == ["a"]
        assert [m["lesson_id"] for m in avoid] == ["b"]

    def test_actions_to_withdraw_respects_mode_and_confidence(self):
        memories = [
            {"avoid": True, "action": "change_sampler",
             "failure_mode": "class_imbalance", "confidence": 0.9},
            {"avoid": True, "action": "change_model",
             "failure_mode": "underfitting", "confidence": 0.9},
            {"avoid": True, "action": "retrain",
             "failure_mode": "class_imbalance", "confidence": 0.2},
            {"avoid": False, "action": "change_image_size",
             "failure_mode": "class_imbalance", "confidence": 0.9},
        ]
        assert actions_to_withdraw(memories, "class_imbalance") == ["change_sampler"]


class TestEmbedder:
    def test_batching_and_ordering(self, make_client):
        batches: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            batches.append(body["input"])
            return embeddings_response([[float(len(t))] for t in body["input"]])

        client = make_client(handler)
        embedder = FireworksEmbedder(client, batch_size=2)
        vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
        assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_cache_avoids_repeat_calls(self, make_client):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body["input"])
            return embeddings_response([[1.0] for _ in body["input"]])

        client = make_client(handler)
        embedder = FireworksEmbedder(client)
        embedder.embed_query("same text")
        embedder.embed_query("same text")
        assert len(calls) == 1

    def test_atlas_index_definition_matches_the_vector_size(self, make_client):
        client = make_client(lambda request: embeddings_response([[0.0] * 1024]))
        embedder = FireworksEmbedder(client, dimensions=1024)
        index = embedder.atlas_index_definition()
        vector_field = index["fields"][0]
        assert vector_field["numDimensions"] == 1024
        assert vector_field["similarity"] == "cosine"
        assert {f["path"] for f in index["fields"][1:]} == {
            "failure_mode", "reusable", "avoid"}

    def test_attach_embeddings_skips_documents_that_already_have_vectors(self, make_client):
        seen: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body["input"])
            return embeddings_response([[0.5] for _ in body["input"]])

        client = make_client(handler)
        embedder = FireworksEmbedder(client)
        docs = attach_embeddings(embedder, [
            {"embedding_text": "new one"},
            {"embedding_text": "old one", "embedding": [9.9]},
        ])
        assert seen == [["new one"]]
        assert docs[0]["embedding"] == [0.5]
        assert docs[1]["embedding"] == [9.9]
        assert docs[0]["embedding_provider"] == "fireworks"


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1], [1, 2])


def test_chat_response_helper_is_used(make_client):
    """Guard so the shared fixture stays exercised if tests are trimmed."""
    client = make_client([chat_response("hi")])
    assert client.chat([{"role": "user", "content": "x"}])["choices"][0][
        "message"]["content"] == "hi"
