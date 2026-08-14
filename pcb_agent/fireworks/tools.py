"""The experiment tool allowlist, as real Fireworks tool definitions.

The project brief is emphatic: "The model should select from executable tools
rather than merely suggesting changes in text." So each action below is a
genuine function-calling tool with a typed parameter schema, and `apply_tool`
turns the model's chosen call into a concrete mutation of the training config.

Two properties matter for the hackathon's headline result:

* Bounded action space. The model cannot invent `change_optimizer`; a run's
  experiment history is therefore comparable across runs, which is what makes
  "5 experiments vs 2 experiments" a meaningful claim.
* Typed, validated arguments. A learning rate of "much lower" is rejected here
  rather than 40 minutes into a training job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import FireworksToolChoiceError
from .schemas import ACTIONS
from .structured import validate

#: Starting point for a PCB defect classifier run. `apply_tool` never mutates
#: the caller's dict; it always returns a copy.
DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "architecture": "resnet18",
    "pretrained": True,
    "learning_rate": 3e-4,
    "lr_schedule": "cosine",
    "batch_size": 32,
    "image_size": 224,
    "epochs": 12,
    "augmentations": ["hflip", "vflip"],
    "aug_strength": 0.3,
    "sampler": "random",
    "oversample_factor": 1.0,
    "class_weights": None,
    "confidence_threshold": 0.5,
}

ARCHITECTURES = [
    "resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3",
    "convnext_tiny", "vit_small_patch16",
]

AUGMENTATIONS = [
    "hflip", "vflip", "rotate90", "random_crop", "color_jitter",
    "gaussian_blur", "random_erasing", "mixup", "cutmix", "scale_jitter",
]


def _tool(name: str, description: str, properties: Mapping[str, Any],
          required: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": dict(properties),
                "required": list(required),
            },
        },
    }


#: Every tool carries a `reason` argument. It costs a few tokens and gives the
#: critic and the MongoDB lesson record the model's stated intent, which is
#: what later runs retrieve. Without it a stored experiment is just a diff.
_REASON = {
    "type": "string",
    "description": "Why this specific change should fix the diagnosed failure.",
}


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "change_model": _tool(
        "change_model",
        "Swap the backbone architecture. Use when the current model is "
        "underfitting or lacks the capacity or resolution for the defect scale.",
        {
            "architecture": {"type": "string", "enum": ARCHITECTURES},
            "pretrained": {"type": "boolean", "default": True},
            "reason": _REASON,
        },
        ["architecture", "reason"],
    ),
    "change_learning_rate": _tool(
        "change_learning_rate",
        "Set the learning rate and optionally the schedule. Use for unstable "
        "or stalled training.",
        {
            "learning_rate": {"type": "number", "minimum": 1e-6, "maximum": 1.0},
            "schedule": {
                "type": "string",
                "enum": ["constant", "cosine", "step", "onecycle"],
            },
            "reason": _REASON,
        },
        ["learning_rate", "reason"],
    ),
    "change_batch_size": _tool(
        "change_batch_size",
        "Set the batch size. Smaller batches give noisier but often better "
        "minority-class gradients; larger batches stabilise training.",
        {
            "batch_size": {"type": "integer", "minimum": 2, "maximum": 512},
            "reason": _REASON,
        },
        ["batch_size", "reason"],
    ),
    "change_image_size": _tool(
        "change_image_size",
        "Set the input resolution. Raise it when defects are small relative to "
        "the board, such as hairline open circuits or missing holes.",
        {
            "image_size": {"type": "integer", "minimum": 64, "maximum": 1024},
            "reason": _REASON,
        },
        ["image_size", "reason"],
    ),
    "change_augmentation": _tool(
        "change_augmentation",
        "Replace the augmentation set and strength. Use against overfitting or "
        "to synthesise variation for rare defect classes.",
        {
            "augmentations": {
                "type": "array",
                "items": {"type": "string", "enum": AUGMENTATIONS},
                "minItems": 0,
                "maxItems": 8,
            },
            "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": _REASON,
        },
        ["augmentations", "reason"],
    ),
    "change_sampler": _tool(
        "change_sampler",
        "Change how training batches are drawn. Weighted sampling is the "
        "standard first move against class imbalance.",
        {
            "strategy": {
                "type": "string",
                "enum": ["random", "weighted", "balanced_batch", "oversample_minority"],
            },
            "oversample_factor": {"type": "number", "minimum": 1.0, "maximum": 20.0},
            "reason": _REASON,
        },
        ["strategy", "reason"],
    ),
    "change_class_weights": _tool(
        "change_class_weights",
        "Reweight the loss per class. Use when recall on a minority class is "
        "the bottleneck and resampling alone was not enough.",
        {
            "mode": {
                "type": "string",
                "enum": ["none", "balanced", "inverse_frequency", "manual"],
            },
            "weights": {
                "type": "object",
                "description": "class name -> weight. Required when mode is 'manual'.",
                "additionalProperties": {"type": "number", "minimum": 0.0},
            },
            "reason": _REASON,
        },
        ["mode", "reason"],
    ),
    "change_confidence_threshold": _tool(
        "change_confidence_threshold",
        "Move the decision threshold. A post-hoc precision/recall trade with "
        "no retraining cost.",
        {
            "threshold": {"type": "number", "minimum": 0.05, "maximum": 0.95},
            "per_class": {
                "type": "object",
                "description": "Optional per-class thresholds overriding the global one.",
                "additionalProperties": {"type": "number", "minimum": 0.05, "maximum": 0.95},
            },
            "reason": _REASON,
        },
        ["threshold", "reason"],
    ),
    "retrain": _tool(
        "retrain",
        "Retrain with the current config unchanged, optionally for longer. Use "
        "only when the evidence says the config is right and training was cut "
        "short. Not a substitute for a real intervention.",
        {
            "epochs": {"type": "integer", "minimum": 1, "maximum": 100},
            "reason": _REASON,
        },
        ["reason"],
    ),
}

assert set(TOOL_SPECS) == set(ACTIONS), "tool allowlist drifted from schemas.ACTIONS"


def tool_definitions(allowed: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The `tools` argument for a chat request.

    Pass `allowed` to narrow the action space mid-run, for example dropping
    `retrain` after it has already been tried, or dropping actions that
    retrieved memory says failed on this exact failure signature.
    """
    if allowed is None:
        return [dict(spec) for spec in TOOL_SPECS.values()]
    unknown = set(allowed) - set(TOOL_SPECS)
    if unknown:
        raise FireworksToolChoiceError(f"unknown tools requested: {sorted(unknown)}")
    return [dict(TOOL_SPECS[name]) for name in allowed]


def force_tool(name: str) -> dict[str, Any]:
    """`tool_choice` value that forces one specific tool."""
    if name not in TOOL_SPECS:
        raise FireworksToolChoiceError(f"unknown tool {name!r}")
    return {"type": "function", "function": {"name": name}}


@dataclass
class ToolCall:
    """A validated tool selection from the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    raw_arguments: str = ""

    @property
    def reason(self) -> str:
        return str(self.arguments.get("reason", ""))

    def as_proposal(self) -> dict[str, Any]:
        """Shape matching `PROPOSAL_SCHEMA.parameters` for the experiment record."""
        params = {k: v for k, v in self.arguments.items() if k != "reason"}
        return {"next_action": self.name, "parameters": params, "rationale": self.reason}


def parse_tool_calls(
    payload: Mapping[str, Any],
    *,
    allowed: Sequence[str] | None = None,
    strict: bool = True,
) -> list[ToolCall]:
    """Extract and validate tool calls from a chat completion response.

    Raises `FireworksToolChoiceError` when the model returned prose instead of
    a tool call, picked a tool outside the allowlist, or produced arguments
    that violate the tool's parameter schema.
    """
    choices = payload.get("choices") or []
    if not choices:
        raise FireworksToolChoiceError("response contained no choices")

    message = choices[0].get("message") or {}
    raw_calls = message.get("tool_calls") or []
    if not raw_calls:
        content = (message.get("content") or "").strip()
        raise FireworksToolChoiceError(
            "model returned no tool call. Set tool_choice='required' to force one. "
            f"Content was: {content[:200]!r}"
        )

    permitted = set(allowed) if allowed is not None else set(TOOL_SPECS)
    parsed: list[ToolCall] = []

    for raw in raw_calls:
        function = raw.get("function") or {}
        name = function.get("name") or ""
        raw_args = function.get("arguments") or "{}"

        if name not in TOOL_SPECS:
            raise FireworksToolChoiceError(
                f"model called unknown tool {name!r}; allowlist is {sorted(TOOL_SPECS)}"
            )
        if name not in permitted:
            raise FireworksToolChoiceError(
                f"model called {name!r}, which was excluded from this step "
                f"(permitted: {sorted(permitted)})"
            )

        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError as exc:
            raise FireworksToolChoiceError(
                f"{name} arguments were not valid JSON: {raw_args[:200]!r}"
            ) from exc

        if strict:
            schema = TOOL_SPECS[name]["function"]["parameters"]
            errors = validate(arguments, schema)
            if errors:
                raise FireworksToolChoiceError(
                    f"{name} arguments failed validation: " + "; ".join(errors[:5])
                )
            _validate_conditionals(name, arguments)

        parsed.append(
            ToolCall(
                name=name,
                arguments=arguments,
                call_id=raw.get("id"),
                raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
            )
        )

    return parsed


def _validate_conditionals(name: str, arguments: Mapping[str, Any]) -> None:
    """Cross-field rules JSON Schema cannot express cheaply."""
    if name == "change_class_weights":
        if arguments.get("mode") == "manual" and not arguments.get("weights"):
            raise FireworksToolChoiceError(
                "change_class_weights(mode='manual') requires a non-empty `weights` map"
            )
    if name == "change_sampler":
        strategy = arguments.get("strategy")
        factor = arguments.get("oversample_factor")
        if strategy == "oversample_minority" and factor is None:
            raise FireworksToolChoiceError(
                "change_sampler(strategy='oversample_minority') requires oversample_factor"
            )


def apply_tool(
    config: Mapping[str, Any], call: ToolCall
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a tool call to a training config.

    Returns `(new_config, diff)` where `diff` maps changed keys to
    `{"from": old, "to": new}`. The diff is what goes into the MongoDB
    experiment record as `intervention`, so a future run can see exactly what
    changed rather than re-deriving it from two full configs.
    """
    new_config = dict(config)
    args = call.arguments

    if call.name == "change_model":
        new_config["architecture"] = args["architecture"]
        if "pretrained" in args:
            new_config["pretrained"] = bool(args["pretrained"])

    elif call.name == "change_learning_rate":
        new_config["learning_rate"] = float(args["learning_rate"])
        if args.get("schedule"):
            new_config["lr_schedule"] = args["schedule"]

    elif call.name == "change_batch_size":
        new_config["batch_size"] = int(args["batch_size"])

    elif call.name == "change_image_size":
        new_config["image_size"] = int(args["image_size"])

    elif call.name == "change_augmentation":
        new_config["augmentations"] = list(args["augmentations"])
        if args.get("strength") is not None:
            new_config["aug_strength"] = float(args["strength"])

    elif call.name == "change_sampler":
        new_config["sampler"] = args["strategy"]
        if args.get("oversample_factor") is not None:
            new_config["oversample_factor"] = float(args["oversample_factor"])
        elif args["strategy"] == "random":
            new_config["oversample_factor"] = 1.0

    elif call.name == "change_class_weights":
        mode = args["mode"]
        new_config["class_weights"] = None if mode == "none" else (
            dict(args["weights"]) if mode == "manual" else mode
        )

    elif call.name == "change_confidence_threshold":
        new_config["confidence_threshold"] = float(args["threshold"])
        if args.get("per_class"):
            new_config["per_class_thresholds"] = dict(args["per_class"])

    elif call.name == "retrain":
        if args.get("epochs") is not None:
            new_config["epochs"] = int(args["epochs"])

    else:  # pragma: no cover - parse_tool_calls rejects unknown names first
        raise FireworksToolChoiceError(f"no handler for tool {call.name!r}")

    return new_config, diff_configs(config, new_config)


def diff_configs(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(before) | set(after)
    return {
        key: {"from": before.get(key), "to": after.get(key)}
        for key in sorted(keys)
        if before.get(key) != after.get(key)
    }


def apply_experiment(
    current_config: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    """The frozen shared interface from the 3-person plan.

    Accepts a plain `{"next_action": ..., "parameters": {...}}` dict, so the
    LangGraph node can call it without importing ToolCall.
    """
    action = proposal.get("next_action")
    if action not in TOOL_SPECS:
        raise FireworksToolChoiceError(
            f"proposal.next_action={action!r} is not in the allowlist {sorted(TOOL_SPECS)}"
        )
    arguments = dict(proposal.get("parameters") or {})
    arguments.setdefault("reason", proposal.get("rationale") or "unspecified")

    call = ToolCall(name=action, arguments=arguments)
    schema = TOOL_SPECS[action]["function"]["parameters"]
    errors = validate(arguments, schema)
    if errors:
        raise FireworksToolChoiceError(
            f"proposal parameters failed {action} schema: " + "; ".join(errors[:5])
        )
    _validate_conditionals(action, arguments)
    new_config, _ = apply_tool(current_config, call)
    return new_config


@dataclass
class ActionHistory:
    """Tracks which (failure_mode, action) pairs have already been tried.

    Feeding this into `tool_definitions(allowed=...)` is the mechanism by which
    retrieved memory actually narrows the action space, instead of just being
    extra prompt text the model may ignore.
    """

    tried: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def record(self, failure_mode: str, action: str, *, helped: bool) -> None:
        self.tried.append((failure_mode, action))
        if not helped:
            self.failed.append((failure_mode, action))

    def failed_actions(self, failure_mode: str) -> list[str]:
        """Actions that did not help this failure mode, within this run."""
        return sorted({action for mode, action in self.failed if mode == failure_mode})

    def allowed_actions(self, failure_mode: str, *, keep_minimum: int = 3) -> list[str]:
        """Drop actions that already failed for this failure mode.

        Never narrows below `keep_minimum` tools, so the model always has a
        real choice and cannot be cornered into a single forced action.
        """
        blocked = {action for mode, action in self.failed if mode == failure_mode}
        remaining = [name for name in TOOL_SPECS if name not in blocked]
        if len(remaining) < keep_minimum:
            return list(TOOL_SPECS)
        return remaining
