"""Model registry for the Fireworks layer.

Fireworks' serverless catalogue rotates: models get promoted, demoted to
dedicated-only, and retired. A hardcoded model id that works today can 404
next month. So this module does three things:

1. Maps *roles* (scientist, critic, vision, embedding, ...) to model ids, so
   the rest of the codebase never hardcodes a model string.
2. Lets every role be overridden by an environment variable.
3. Ships `validate_registry()`, which checks the configured ids against the
   live `GET /v1/models` catalogue and reports which ones are dead.

Run `python -m scripts.fireworks_smoke --models` to see the current state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

_PREFIX = "accounts/fireworks/models/"


class Role(str, Enum):
    """What we need a model *for*, decoupled from which model that is."""

    SCIENTIST = "scientist"      # diagnosis + experiment proposal, tool calling
    CRITIC = "critic"            # before/after judgement + lesson writing
    REASONER = "reasoner"        # optional deep-thinking escalation path
    FAST = "fast"                # cheap summarisation, failure signatures, titles
    VISION = "vision"            # PCB crop inspection
    VISION_SMALL = "vision_small"
    EMBEDDING = "embedding"      # Atlas Vector Search vectors
    RERANK = "rerank"            # precision pass over vector-search recall


#: Defaults verified against the Fireworks serverless catalogue in Aug 2026.
#: Override any of these with FIREWORKS_MODEL_<ROLE>, e.g.
#: FIREWORKS_MODEL_SCIENTIST=accounts/fireworks/models/deepseek-v3p1
DEFAULT_MODELS: dict[Role, str] = {
    Role.SCIENTIST: _PREFIX + "kimi-k2-instruct-0905",
    Role.CRITIC: _PREFIX + "deepseek-v3p1",
    Role.REASONER: _PREFIX + "kimi-k2-thinking",
    Role.FAST: _PREFIX + "llama-v3p3-70b-instruct",
    Role.VISION: _PREFIX + "qwen3-vl-235b-a22b-instruct",
    Role.VISION_SMALL: _PREFIX + "qwen2p5-vl-32b-instruct",
    Role.EMBEDDING: _PREFIX + "qwen3-embedding-8b",
    Role.RERANK: _PREFIX + "qwen3-reranker-8b",
}

#: Ordered fallbacks tried when the primary id is unavailable. Keep these to
#: models in the same capability class, not just "any model that responds".
FALLBACK_MODELS: dict[Role, tuple[str, ...]] = {
    Role.SCIENTIST: (
        _PREFIX + "deepseek-v3p1",
        _PREFIX + "qwen3-235b-a22b-instruct-2507",
        _PREFIX + "gpt-oss-120b",
        _PREFIX + "llama-v3p3-70b-instruct",
    ),
    Role.CRITIC: (
        _PREFIX + "kimi-k2-instruct-0905",
        _PREFIX + "qwen3-235b-a22b-instruct-2507",
        _PREFIX + "llama-v3p3-70b-instruct",
    ),
    Role.REASONER: (
        _PREFIX + "deepseek-r1-0528",
        _PREFIX + "qwq-32b",
    ),
    Role.FAST: (
        _PREFIX + "gpt-oss-20b",
        _PREFIX + "qwen3-8b",
    ),
    Role.VISION: (
        _PREFIX + "qwen3-vl-30b-a3b-instruct",
        _PREFIX + "qwen2p5-vl-72b-instruct",
    ),
    Role.VISION_SMALL: (
        _PREFIX + "qwen2p5-vl-7b-instruct",
        _PREFIX + "qwen3-vl-8b-instruct",
    ),
    Role.EMBEDDING: (
        _PREFIX + "qwen3-embedding-4b",
        "nomic-ai/nomic-embed-text-v1.5",
    ),
    Role.RERANK: (
        _PREFIX + "qwen3-reranker-4b",
        _PREFIX + "qwen3-reranker-0p6b",
    ),
}

#: Roles that must not be swapped silently. Changing the embedding model
#: invalidates every vector already written to Atlas, so we refuse to fall
#: back automatically and make the operator do it on purpose.
PINNED_ROLES = frozenset({Role.EMBEDDING})


def env_var_for(role: Role) -> str:
    return f"FIREWORKS_MODEL_{role.name}"


def resolve(role: Role, env: dict[str, str] | None = None) -> str:
    """Return the model id for `role`, honouring the environment override."""
    source = os.environ if env is None else env
    override = source.get(env_var_for(role))
    if override and override.strip():
        return override.strip()
    return DEFAULT_MODELS[role]


def alternate_id(model_id: str) -> str | None:
    """Fireworks accepts both `accounts/fireworks/models/x` and, for some
    first-party models, the short `fireworks/x`. The docs are inconsistent
    about which form each endpoint wants, so we can retry once with the other
    spelling before giving up on a 404.
    """
    if model_id.startswith(_PREFIX):
        return "fireworks/" + model_id[len(_PREFIX):]
    if model_id.startswith("fireworks/"):
        return _PREFIX + model_id[len("fireworks/"):]
    return None


def short_name(model_id: str) -> str:
    """`accounts/fireworks/models/kimi-k2-instruct-0905` -> `kimi-k2-instruct-0905`."""
    return model_id.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class RegistryCheck:
    """Result of comparing the configured roster against the live catalogue."""

    available: dict[Role, str] = field(default_factory=dict)
    missing: dict[Role, str] = field(default_factory=dict)
    repaired: dict[Role, tuple[str, str]] = field(default_factory=dict)
    catalogue_size: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing

    def report(self) -> str:
        lines = [f"Fireworks catalogue: {self.catalogue_size} models visible"]
        for role, mid in sorted(self.available.items(), key=lambda kv: kv[0].value):
            lines.append(f"  ok       {role.value:<13} {mid}")
        for role, (old, new) in sorted(self.repaired.items(), key=lambda kv: kv[0].value):
            lines.append(f"  fallback {role.value:<13} {old} -> {new}")
        for role, mid in sorted(self.missing.items(), key=lambda kv: kv[0].value):
            lines.append(f"  DEAD     {role.value:<13} {mid}")
        if self.missing:
            lines.append("")
            lines.append("Set the env overrides listed in FIREWORKS.md for the DEAD roles,")
            lines.append("or browse https://app.fireworks.ai/models for a live replacement.")
        return "\n".join(lines)


def validate_registry(
    catalogue: Iterable[str],
    roles: Iterable[Role] | None = None,
    *,
    repair: bool = True,
    env: dict[str, str] | None = None,
) -> RegistryCheck:
    """Check configured model ids against `catalogue` (ids from GET /v1/models).

    With `repair=True` a role whose primary id is gone is reassigned to the
    first live entry in FALLBACK_MODELS, except for PINNED_ROLES.
    """
    known = set(catalogue)
    check_roles = list(roles) if roles is not None else list(Role)

    available: dict[Role, str] = {}
    missing: dict[Role, str] = {}
    repaired: dict[Role, tuple[str, str]] = {}

    for role in check_roles:
        primary = resolve(role, env)
        if _in_catalogue(primary, known):
            available[role] = primary
            continue
        if repair and role not in PINNED_ROLES:
            replacement = next(
                (cand for cand in FALLBACK_MODELS.get(role, ()) if _in_catalogue(cand, known)),
                None,
            )
            if replacement is not None:
                repaired[role] = (primary, replacement)
                available[role] = replacement
                continue
        missing[role] = primary

    return RegistryCheck(
        available=available,
        missing=missing,
        repaired=repaired,
        catalogue_size=len(known),
    )


def _in_catalogue(model_id: str, known: set[str]) -> bool:
    if model_id in known:
        return True
    alt = alternate_id(model_id)
    if alt and alt in known:
        return True
    # Some catalogue listings return bare ids without the account prefix.
    return short_name(model_id) in known
