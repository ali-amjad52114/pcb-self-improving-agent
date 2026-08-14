"""Structured output: response_format assembly, parsing, validation, repair."""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import chat_response, json_chat_response

from pcb_agent.fireworks.errors import FireworksStructuredOutputError
from pcb_agent.fireworks.schemas import ALL_SCHEMAS, DIAGNOSIS_SCHEMA
from pcb_agent.fireworks.structured import (
    complete_json,
    grammar_response_format,
    json_object_response_format,
    json_schema_response_format,
    parse_json,
    schema_instruction,
    validate,
)

SIMPLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "score"],
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


class TestResponseFormat:
    def test_json_schema_shape_matches_the_fireworks_docs(self):
        fmt = json_schema_response_format("Diagnosis", DIAGNOSIS_SCHEMA)
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "Diagnosis"
        assert fmt["json_schema"]["schema"]["type"] == "object"

    def test_json_object_and_grammar_modes(self):
        assert json_object_response_format() == {"type": "json_object"}
        fmt = grammar_response_format("root ::= \"a\"")
        assert fmt["type"] == "grammar"
        assert "root" in fmt["grammar"]

    def test_schema_instruction_carries_the_schema_and_forbids_prose(self):
        text = schema_instruction("Thing", SIMPLE_SCHEMA)
        assert "single JSON object" in text
        assert json.dumps(SIMPLE_SCHEMA, indent=2) in text


class TestParseJson:
    def test_plain_json(self):
        assert parse_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_prose(self):
        assert parse_json('Sure, here it is: {"a": 1} Hope that helps!') == {"a": 1}

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_json("   ")

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_json("definitely not json")


class TestValidate:
    def test_valid_instance_has_no_errors(self):
        assert validate({"name": "x", "score": 0.5}, SIMPLE_SCHEMA) == []

    def test_missing_required(self):
        errors = validate({"name": "x"}, SIMPLE_SCHEMA)
        assert any("score" in e for e in errors)

    def test_wrong_type(self):
        errors = validate({"name": 1, "score": 0.5}, SIMPLE_SCHEMA)
        assert any("name" in e for e in errors)

    def test_out_of_range(self):
        errors = validate({"name": "x", "score": 5}, SIMPLE_SCHEMA)
        assert errors

    def test_additional_property_rejected(self):
        errors = validate({"name": "x", "score": 0.5, "extra": True}, SIMPLE_SCHEMA)
        assert any("extra" in e for e in errors)

    def test_enum_is_enforced(self):
        errors = validate(
            {
                "diagnosis": "d", "hypothesis": "h", "confidence": 0.5,
                "failure_signature": {
                    "summary": "s", "affected_classes": [],
                    "failure_mode": "not_a_real_mode",
                },
            },
            DIAGNOSIS_SCHEMA,
        )
        assert any("failure_mode" in e or "not_a_real_mode" in e for e in errors)

    def test_bool_does_not_satisfy_number(self):
        assert validate({"name": "x", "score": True}, SIMPLE_SCHEMA)

    @pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
    def test_every_shipped_schema_is_well_formed(self, name):
        schema = ALL_SCHEMAS[name]
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
        assert schema["required"]
        for key in schema["required"]:
            assert key in schema["properties"], f"{name}.{key} required but undefined"


class TestCompleteJson:
    def _valid_diagnosis(self) -> dict:
        return {
            "diagnosis": "The minority open_circuit class is under-recalled.",
            "failure_signature": {
                "summary": "Low recall on a rare defect class under heavy imbalance.",
                "affected_classes": ["open_circuit"],
                "failure_mode": "class_imbalance",
            },
            "hypothesis": "Weighted sampling will raise minority recall.",
            "confidence": 0.72,
        }

    def test_returns_the_validated_object(self, make_client):
        client = make_client([json_chat_response(self._valid_diagnosis())])
        result = complete_json(
            client, name="Diagnosis", schema=DIAGNOSIS_SCHEMA,
            messages=[{"role": "user", "content": "diagnose"}],
        )
        assert result["failure_signature"]["failure_mode"] == "class_imbalance"

    def test_schema_is_sent_in_both_places(self, make_client):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return json_chat_response(self._valid_diagnosis())

        client = make_client(handler)
        complete_json(client, name="Diagnosis", schema=DIAGNOSIS_SCHEMA,
                      messages=[{"role": "user", "content": "diagnose"}])

        assert captured["response_format"]["type"] == "json_schema"
        last_user = captured["messages"][-1]["content"]
        assert "JSON Schema named Diagnosis" in last_user

    def test_invalid_output_is_repaired_on_retry(self, make_client):
        client = make_client([
            json_chat_response({"diagnosis": "incomplete"}),
            json_chat_response(self._valid_diagnosis()),
        ])
        result = complete_json(
            client, name="Diagnosis", schema=DIAGNOSIS_SCHEMA,
            messages=[{"role": "user", "content": "diagnose"}],
        )
        assert result["confidence"] == 0.72

    def test_repair_prompt_lists_the_validation_errors(self, make_client):
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            if len(seen) == 1:
                return json_chat_response({"diagnosis": "incomplete"})
            return json_chat_response(self._valid_diagnosis())

        client = make_client(handler)
        complete_json(client, name="Diagnosis", schema=DIAGNOSIS_SCHEMA,
                      messages=[{"role": "user", "content": "diagnose"}])

        repair_turn = seen[1]["messages"][-1]["content"]
        assert "did not satisfy the schema" in repair_turn
        assert "hypothesis" in repair_turn

    def test_exhausting_attempts_raises(self, make_client):
        client = make_client(lambda request: json_chat_response({"nope": 1}))
        with pytest.raises(FireworksStructuredOutputError) as excinfo:
            complete_json(client, name="Diagnosis", schema=DIAGNOSIS_SCHEMA,
                          messages=[{"role": "user", "content": "d"}])
        assert excinfo.value.attempts == 2  # structured_max_attempts in the fixture

    def test_truncation_fails_fast_with_an_actionable_message(self, make_client):
        client = make_client([chat_response('{"diagnosis": "trunc',
                                            finish_reason="length")])
        with pytest.raises(FireworksStructuredOutputError, match="max_tokens"):
            complete_json(client, name="Diagnosis", schema=DIAGNOSIS_SCHEMA,
                          messages=[{"role": "user", "content": "d"}])
