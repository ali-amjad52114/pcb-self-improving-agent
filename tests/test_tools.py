"""The experiment tool allowlist: definitions, parsing, validation, application."""

from __future__ import annotations

import pytest
from conftest import tool_call

from pcb_agent.fireworks.errors import FireworksToolChoiceError
from pcb_agent.fireworks.schemas import ACTIONS
from pcb_agent.fireworks.tools import (
    DEFAULT_TRAINING_CONFIG,
    TOOL_SPECS,
    ActionHistory,
    ToolCall,
    apply_experiment,
    apply_tool,
    diff_configs,
    force_tool,
    parse_tool_calls,
    tool_definitions,
)


def _payload(*calls):
    return {"choices": [{"message": {"role": "assistant", "content": None,
                                     "tool_calls": list(calls)}}]}


def test_allowlist_matches_the_brief():
    assert set(TOOL_SPECS) == set(ACTIONS)
    assert len(TOOL_SPECS) == 9


def test_tool_definitions_shape_is_openai_compatible():
    for spec in tool_definitions():
        assert spec["type"] == "function"
        function = spec["function"]
        assert function["name"] in TOOL_SPECS
        assert function["description"]
        params = function["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert "reason" in params["properties"]
        # Fireworks caps tool names at 64 chars.
        assert len(function["name"]) <= 64


def test_tool_definitions_can_be_narrowed():
    subset = tool_definitions(["change_sampler", "change_image_size"])
    assert [s["function"]["name"] for s in subset] == [
        "change_sampler", "change_image_size"]

    with pytest.raises(FireworksToolChoiceError):
        tool_definitions(["change_optimizer"])


def test_force_tool():
    assert force_tool("retrain") == {"type": "function",
                                     "function": {"name": "retrain"}}
    with pytest.raises(FireworksToolChoiceError):
        force_tool("nope")


def test_parse_tool_calls_happy_path():
    payload = _payload(tool_call("change_sampler",
                                 {"strategy": "weighted", "reason": "imbalance"}))
    calls = parse_tool_calls(payload)
    assert len(calls) == 1
    assert calls[0].name == "change_sampler"
    assert calls[0].reason == "imbalance"


def test_parse_tool_calls_rejects_prose():
    payload = {"choices": [{"message": {"role": "assistant",
                                        "content": "I think you should use weighted sampling."}}]}
    with pytest.raises(FireworksToolChoiceError, match="no tool call"):
        parse_tool_calls(payload)


def test_parse_tool_calls_rejects_unknown_tool():
    payload = _payload(tool_call("change_optimizer", {"reason": "x"}))
    with pytest.raises(FireworksToolChoiceError, match="unknown tool"):
        parse_tool_calls(payload)


def test_parse_tool_calls_enforces_the_narrowed_allowlist():
    payload = _payload(tool_call("retrain", {"reason": "again"}))
    with pytest.raises(FireworksToolChoiceError, match="excluded"):
        parse_tool_calls(payload, allowed=["change_sampler"])


def test_parse_tool_calls_validates_arguments_against_the_schema():
    payload = _payload(tool_call("change_learning_rate",
                                 {"learning_rate": 42.0, "reason": "why not"}))
    with pytest.raises(FireworksToolChoiceError, match="failed validation"):
        parse_tool_calls(payload)


def test_parse_tool_calls_rejects_missing_required_argument():
    payload = _payload(tool_call("change_image_size", {"reason": "small defects"}))
    with pytest.raises(FireworksToolChoiceError, match="image_size"):
        parse_tool_calls(payload)


def test_parse_tool_calls_rejects_malformed_json_arguments():
    payload = _payload({"id": "c1", "type": "function",
                        "function": {"name": "retrain", "arguments": "{not json"}})
    with pytest.raises(FireworksToolChoiceError, match="not valid JSON"):
        parse_tool_calls(payload)


def test_cross_field_rule_manual_weights():
    payload = _payload(tool_call("change_class_weights",
                                 {"mode": "manual", "reason": "minority"}))
    with pytest.raises(FireworksToolChoiceError, match="weights"):
        parse_tool_calls(payload)


def test_cross_field_rule_oversample_factor():
    payload = _payload(tool_call("change_sampler",
                                 {"strategy": "oversample_minority", "reason": "r"}))
    with pytest.raises(FireworksToolChoiceError, match="oversample_factor"):
        parse_tool_calls(payload)


@pytest.mark.parametrize(
    "name,args,key,expected",
    [
        ("change_model", {"architecture": "resnet50"}, "architecture", "resnet50"),
        ("change_learning_rate", {"learning_rate": 1e-4}, "learning_rate", 1e-4),
        ("change_batch_size", {"batch_size": 64}, "batch_size", 64),
        ("change_image_size", {"image_size": 384}, "image_size", 384),
        ("change_sampler", {"strategy": "weighted"}, "sampler", "weighted"),
        ("change_confidence_threshold", {"threshold": 0.3},
         "confidence_threshold", 0.3),
        ("retrain", {"epochs": 30}, "epochs", 30),
    ],
)
def test_apply_tool_mutates_the_right_key(name, args, key, expected):
    call = ToolCall(name=name, arguments={**args, "reason": "test"})
    new_config, diff = apply_tool(DEFAULT_TRAINING_CONFIG, call)
    assert new_config[key] == expected
    assert key in diff
    assert diff[key]["to"] == expected
    # The input must never be mutated in place.
    assert DEFAULT_TRAINING_CONFIG[key] != expected or diff == {}


def test_apply_tool_class_weights_modes():
    base = DEFAULT_TRAINING_CONFIG
    for mode, expected in [("none", None), ("balanced", "balanced"),
                           ("inverse_frequency", "inverse_frequency")]:
        call = ToolCall("change_class_weights", {"mode": mode, "reason": "r"})
        assert apply_tool(base, call)[0]["class_weights"] == expected

    manual = ToolCall("change_class_weights",
                      {"mode": "manual", "weights": {"open_circuit": 4.0},
                       "reason": "r"})
    assert apply_tool(base, manual)[0]["class_weights"] == {"open_circuit": 4.0}


def test_apply_tool_augmentation_replaces_the_set():
    call = ToolCall("change_augmentation",
                    {"augmentations": ["mixup", "random_erasing"], "strength": 0.6,
                     "reason": "rare classes"})
    new_config, _ = apply_tool(DEFAULT_TRAINING_CONFIG, call)
    assert new_config["augmentations"] == ["mixup", "random_erasing"]
    assert new_config["aug_strength"] == 0.6


def test_diff_configs_reports_only_changes():
    before = {"a": 1, "b": 2}
    after = {"a": 1, "b": 3, "c": 4}
    assert diff_configs(before, after) == {
        "b": {"from": 2, "to": 3},
        "c": {"from": None, "to": 4},
    }


def test_apply_experiment_frozen_interface():
    proposal = {
        "next_action": "change_sampler",
        "parameters": {"strategy": "weighted"},
        "rationale": "minority recall is the bottleneck",
    }
    new_config = apply_experiment(DEFAULT_TRAINING_CONFIG, proposal)
    assert new_config["sampler"] == "weighted"
    assert DEFAULT_TRAINING_CONFIG["sampler"] == "random"


def test_apply_experiment_rejects_out_of_allowlist_actions():
    with pytest.raises(FireworksToolChoiceError, match="not in the allowlist"):
        apply_experiment(DEFAULT_TRAINING_CONFIG,
                         {"next_action": "delete_dataset", "parameters": {}})


def test_apply_experiment_rejects_bad_parameters():
    with pytest.raises(FireworksToolChoiceError, match="failed change_batch_size schema"):
        apply_experiment(DEFAULT_TRAINING_CONFIG,
                         {"next_action": "change_batch_size",
                          "parameters": {"batch_size": 100000}})


class TestActionHistory:
    def test_failed_actions_are_scoped_to_the_failure_mode(self):
        history = ActionHistory()
        history.record("class_imbalance", "change_sampler", helped=False)
        history.record("small_object_detection", "change_image_size", helped=True)

        assert history.failed_actions("class_imbalance") == ["change_sampler"]
        assert history.failed_actions("small_object_detection") == []

    def test_allowed_actions_drops_known_failures(self):
        history = ActionHistory()
        history.record("class_imbalance", "change_sampler", helped=False)
        allowed = history.allowed_actions("class_imbalance")
        assert "change_sampler" not in allowed
        assert "change_class_weights" in allowed

    def test_allowed_actions_never_corners_the_model(self):
        history = ActionHistory()
        for action in list(TOOL_SPECS)[:-1]:
            history.record("class_imbalance", action, helped=False)
        # Only one action would remain, so the full list is restored.
        assert set(history.allowed_actions("class_imbalance")) == set(TOOL_SPECS)
