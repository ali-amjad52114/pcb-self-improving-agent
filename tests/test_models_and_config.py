"""Model registry, settings resolution, and the dependency-free validator."""

from __future__ import annotations

import pytest

from pcb_agent.fireworks.config import FireworksSettings
from pcb_agent.fireworks.errors import FireworksConfigError
from pcb_agent.fireworks.models import (
    DEFAULT_MODELS,
    FALLBACK_MODELS,
    PINNED_ROLES,
    Role,
    alternate_id,
    env_var_for,
    resolve,
    short_name,
    validate_registry,
)
from pcb_agent.fireworks.structured import _validate_subset


class TestRegistry:
    def test_every_role_has_a_default(self):
        assert set(DEFAULT_MODELS) == set(Role)

    def test_env_override_wins(self):
        env = {env_var_for(Role.SCIENTIST): "accounts/me/models/custom"}
        assert resolve(Role.SCIENTIST, env) == "accounts/me/models/custom"

    def test_blank_env_override_is_ignored(self):
        assert resolve(Role.SCIENTIST, {env_var_for(Role.SCIENTIST): "  "}) == \
            DEFAULT_MODELS[Role.SCIENTIST]

    def test_alternate_id_round_trips(self):
        full = "accounts/fireworks/models/kimi-k2-instruct-0905"
        short = "fireworks/kimi-k2-instruct-0905"
        assert alternate_id(full) == short
        assert alternate_id(short) == full
        assert alternate_id("nomic-ai/nomic-embed-text-v1.5") is None

    def test_short_name(self):
        assert short_name("accounts/fireworks/models/glm-4p6") == "glm-4p6"

    def test_all_roles_available(self):
        check = validate_registry(DEFAULT_MODELS.values())
        assert check.ok
        assert not check.repaired
        assert len(check.available) == len(Role)

    def test_missing_model_is_repaired_from_the_fallback_list(self):
        catalogue = set(DEFAULT_MODELS.values()) - {DEFAULT_MODELS[Role.SCIENTIST]}
        catalogue.add(FALLBACK_MODELS[Role.SCIENTIST][0])

        check = validate_registry(catalogue, roles=[Role.SCIENTIST])
        assert check.ok
        assert check.repaired[Role.SCIENTIST][1] == FALLBACK_MODELS[Role.SCIENTIST][0]

    def test_embedding_role_is_never_silently_swapped(self):
        assert Role.EMBEDDING in PINNED_ROLES
        catalogue = set(FALLBACK_MODELS[Role.EMBEDDING])
        check = validate_registry(catalogue, roles=[Role.EMBEDDING])
        assert not check.ok
        assert Role.EMBEDDING in check.missing

    def test_dead_model_with_no_live_fallback_is_reported(self):
        check = validate_registry(["accounts/fireworks/models/something-else"],
                                  roles=[Role.VISION])
        assert not check.ok
        assert "DEAD" in check.report()
        assert "FIREWORKS.md" in check.report()

    def test_catalogue_matching_tolerates_both_id_spellings(self):
        catalogue = [alternate_id(DEFAULT_MODELS[Role.FAST])]
        assert validate_registry(catalogue, roles=[Role.FAST]).ok


class TestSettings:
    def test_missing_key_raises_an_actionable_error(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        with pytest.raises(FireworksConfigError, match=".env.example"):
            FireworksSettings.from_env()

    def test_require_key_false_allows_inspection(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        settings = FireworksSettings.from_env(require_key=False)
        assert settings.api_key == ""
        assert settings.models[Role.SCIENTIST] == DEFAULT_MODELS[Role.SCIENTIST]

    def test_env_is_parsed(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_abc")
        monkeypatch.setenv("FIREWORKS_TEMPERATURE", "0.9")
        monkeypatch.setenv("FIREWORKS_MAX_TOKENS", "512")
        monkeypatch.setenv("FIREWORKS_PERF_METRICS", "true")
        monkeypatch.setenv("FIREWORKS_EMBEDDING_DIMENSIONS", "none")
        settings = FireworksSettings.from_env()
        assert settings.temperature == 0.9
        assert settings.max_tokens == 512
        assert settings.perf_metrics is True
        assert settings.embedding_dimensions is None

    def test_bad_numeric_env_is_rejected(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_abc")
        monkeypatch.setenv("FIREWORKS_MAX_TOKENS", "lots")
        with pytest.raises(FireworksConfigError, match="FIREWORKS_MAX_TOKENS"):
            FireworksSettings.from_env()

    def test_base_url_is_normalised_and_checked(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_abc")
        monkeypatch.setenv("FIREWORKS_BASE_URL",
                           "https://api.fireworks.ai/inference/v1/")
        assert FireworksSettings.from_env().base_url.endswith("/inference/v1")

        monkeypatch.setenv("FIREWORKS_BASE_URL", "ftp://nope")
        with pytest.raises(FireworksConfigError, match="BASE_URL"):
            FireworksSettings.from_env()

    def test_redacted_never_leaks_the_key(self):
        settings = FireworksSettings(api_key="fw_supersecretvalue",
                                     models=dict(DEFAULT_MODELS))
        redacted = settings.redacted()
        assert "supersecret" not in redacted["api_key"]
        assert redacted["api_key"].startswith("fw_s")

    def test_with_model_returns_a_copy(self):
        settings = FireworksSettings(api_key="k", models=dict(DEFAULT_MODELS))
        updated = settings.with_model(Role.FAST, "accounts/x/models/y")
        assert updated.model_for(Role.FAST) == "accounts/x/models/y"
        assert settings.model_for(Role.FAST) == DEFAULT_MODELS[Role.FAST]


class TestFallbackValidator:
    """The built-in validator used when `jsonschema` is not installed."""

    SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a"],
        "properties": {
            "a": {"type": "string"},
            "n": {"type": "number", "minimum": 0, "maximum": 1},
            "items": {"type": "array", "items": {"type": "integer"}},
            "nested": {
                "type": "object",
                "required": ["x"],
                "properties": {"x": {"type": "string", "enum": ["p", "q"]}},
            },
        },
    }

    def test_valid(self):
        assert _validate_subset(
            {"a": "s", "n": 0.5, "items": [1, 2], "nested": {"x": "p"}},
            self.SCHEMA, path="$",
        ) == []

    def test_missing_required(self):
        assert _validate_subset({}, self.SCHEMA, path="$")

    def test_additional_property(self):
        errors = _validate_subset({"a": "s", "zzz": 1}, self.SCHEMA, path="$")
        assert any("zzz" in e for e in errors)

    def test_range(self):
        assert _validate_subset({"a": "s", "n": 2}, self.SCHEMA, path="$")

    def test_nested_enum(self):
        errors = _validate_subset({"a": "s", "nested": {"x": "z"}},
                                  self.SCHEMA, path="$")
        assert any("nested.x" in e for e in errors)

    def test_array_items(self):
        errors = _validate_subset({"a": "s", "items": [1, "two"]},
                                  self.SCHEMA, path="$")
        assert any("items[1]" in e for e in errors)

    def test_bool_is_not_an_integer(self):
        errors = _validate_subset({"a": "s", "items": [True]}, self.SCHEMA, path="$")
        assert errors
