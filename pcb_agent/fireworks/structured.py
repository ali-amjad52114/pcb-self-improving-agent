"""Schema-constrained generation on top of `FireworksClient`.

Fireworks constrains decoding to a JSON Schema, but the docs are explicit
about two footguns that bite in practice:

1. "Include the schema in both your prompt and the `response_format`" - the
   model is not shown the schema automatically, and without it the output is
   valid-but-wrong JSON.
2. Without an explicit instruction to produce JSON, "the model may generate
   whitespace indefinitely until hitting token limits".

`complete_json` handles both, then validates the parsed object and, on
failure, re-asks with the validation errors appended. That repair loop is what
keeps the LangGraph scientist node from crashing a 4-hour run over one bad
generation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Sequence

from .client import FireworksClient
from .errors import FireworksStructuredOutputError
from .models import Role

log = logging.getLogger("pcb_agent.fireworks.structured")

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def json_schema_response_format(name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Build the `response_format` value for Fireworks JSON-schema mode."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": dict(schema)}}


def json_object_response_format() -> dict[str, Any]:
    """Loose JSON mode: valid JSON, no schema enforcement."""
    return {"type": "json_object"}


def grammar_response_format(grammar: str) -> dict[str, Any]:
    """Fireworks BNF grammar mode.

    Useful when the output is not JSON at all, for example forcing a bare
    verdict token. See `VERDICT_GRAMMAR` below.
    """
    return {"type": "grammar", "grammar": grammar}


#: Example grammar: forces exactly one of four verdict words and nothing else.
#: Cheaper and more reliable than asking for a JSON object with one field.
VERDICT_GRAMMAR = """
root ::= verdict
verdict ::= "improved" | "regressed" | "no_change" | "mixed"
"""


def schema_instruction(name: str, schema: Mapping[str, Any]) -> str:
    """The schema text appended to the prompt, per the Fireworks docs."""
    return (
        f"Respond with a single JSON object and nothing else. No prose, no code "
        f"fence, no trailing commentary. It must validate against this "
        f"JSON Schema named {name}:\n\n"
        f"{json.dumps(schema, indent=2)}"
    )


def complete_json(
    client: FireworksClient,
    *,
    name: str,
    schema: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    role: Role = Role.SCIENTIST,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    purpose: str | None = None,
    max_attempts: int | None = None,
    inject_schema: bool = True,
) -> dict[str, Any]:
    """Generate an object that validates against `schema`.

    Retries with the validation errors fed back to the model, up to
    `settings.structured_max_attempts`.
    """
    attempts = max_attempts or client.settings.structured_max_attempts
    convo: list[dict[str, Any]] = [dict(m) for m in messages]

    if inject_schema:
        convo = _append_to_last_user(convo, "\n\n" + schema_instruction(name, schema))

    response_format = json_schema_response_format(name, schema)
    last_raw: str | None = None
    last_errors: list[str] = []

    for attempt in range(1, attempts + 1):
        payload = client.chat(
            convo,
            role=role,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            purpose=purpose or f"json:{name}",
        )
        choice = (payload.get("choices") or [{}])[0]
        finish = choice.get("finish_reason")
        raw = (choice.get("message") or {}).get("content") or ""
        last_raw = raw

        if finish == "length":
            # Truncated JSON is unparseable by definition; the docs say raise
            # max_tokens rather than retry at the same budget.
            raise FireworksStructuredOutputError(
                f"{name} generation hit the token limit and was truncated. "
                f"Raise max_tokens (currently "
                f"{max_tokens or client.settings.max_tokens}).",
                raw=raw,
                attempts=attempt,
            )

        try:
            parsed = parse_json(raw)
        except ValueError as exc:
            last_errors = [str(exc)]
        else:
            errors = validate(parsed, schema)
            if not errors:
                return parsed
            last_errors = errors

        log.warning("%s attempt %d/%d failed validation: %s",
                    name, attempt, attempts, "; ".join(last_errors[:3]))
        if attempt < attempts:
            convo = convo + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "That response did not satisfy the schema. Problems:\n"
                        + "\n".join(f"- {e}" for e in last_errors[:10])
                        + "\n\nReturn the corrected JSON object only."
                    ),
                },
            ]

    raise FireworksStructuredOutputError(
        f"{name} did not validate after {attempts} attempts: "
        + "; ".join(last_errors[:5]),
        raw=last_raw,
        attempts=attempts,
    )


def parse_json(raw: str) -> Any:
    """Parse model output that should be JSON, tolerating a stray code fence."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("model returned an empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = _JSON_BLOCK.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Last resort: the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"response was not valid JSON: {text[:200]!r}")


# --------------------------------------------------------------- validation


def validate(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors, empty when valid.

    Uses `jsonschema` when installed for full JSON Schema coverage, and falls
    back to a built-in checker for the subset these schemas use. The fallback
    exists so the package has no hard dependency beyond httpx.
    """
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:
        return _validate_subset(instance, schema, path="$")

    validator = jsonschema.Draft202012Validator(dict(schema))
    return [
        f"{_pointer(err.absolute_path)}: {err.message}"
        for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    ]


def _pointer(parts: Any) -> str:
    rendered = "".join(f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in parts)
    return f"${rendered}" if rendered else "$"


_TYPES: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def _validate_subset(instance: Any, schema: Mapping[str, Any], *, path: str) -> list[str]:
    """Minimal validator: type, required, properties, additionalProperties,
    enum, items, minimum/maximum. Enough for every schema in schemas.py.
    """
    errors: list[str] = []

    expected = schema.get("type")
    if expected:
        py_type = _TYPES.get(expected)
        # bool is a subclass of int in Python; do not let True satisfy integer.
        if py_type is not None:
            bad_bool = expected in ("integer", "number") and isinstance(instance, bool)
            if bad_bool or not isinstance(instance, py_type):
                errors.append(f"{path}: expected {expected}, got "
                              f"{type(instance).__name__}")
                return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} is below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} is above maximum {schema['maximum']}")

    if isinstance(instance, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{path}: unexpected property {key!r}")
        for key, sub_schema in properties.items():
            if key in instance:
                errors.extend(
                    _validate_subset(instance[key], sub_schema, path=f"{path}.{key}")
                )

    if isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(instance):
                errors.extend(
                    _validate_subset(item, item_schema, path=f"{path}[{index}]")
                )

    return errors


def _append_to_last_user(
    messages: list[dict[str, Any]], suffix: str
) -> list[dict[str, Any]]:
    """Append schema text to the final user turn, or add one if there is none."""
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"] = message["content"] + suffix
            return messages
    messages.append({"role": "user", "content": suffix.strip()})
    return messages
