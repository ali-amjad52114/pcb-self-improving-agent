"""Fireworks AI integration for the PCB self-improving agent.

Fireworks is the primary intelligence layer for this project. Everything the
agent decides goes through here:

    scientist   diagnose a failure, choose one executable experiment
    critic      judge the outcome, write the reusable lesson
    embeddings  vectors for MongoDB Atlas Vector Search
    rerank      precision pass over vector-search recall
    vision      inspect the PCB crops the classifier got wrong

See FIREWORKS.md for setup, the model registry, cost control, and the mapping
from each Fireworks capability to the part of the agent that uses it.

Quick start:

    from pcb_agent.fireworks import FireworksClient, diagnose, propose_experiment

    with FireworksClient() as client:
        d = diagnose(client, state, memories)
        p = propose_experiment(client, state, d, memories=memories)
        print(client.ledger.summary_line())
"""

from __future__ import annotations

from .client import FireworksClient, build_chat_body
from .config import DEFAULT_BASE_URL, FireworksSettings
from .critic import critique_result, embedding_text, to_lesson_document
from .embeddings import (
    FireworksEmbedder,
    LangChainFireworksEmbeddings,
    attach_embeddings,
    cosine_similarity,
)
from .errors import (
    FireworksAuthError,
    FireworksBadRequestError,
    FireworksConfigError,
    FireworksError,
    FireworksNotFoundError,
    FireworksRateLimitError,
    FireworksServerError,
    FireworksStructuredOutputError,
    FireworksTimeoutError,
    FireworksToolChoiceError,
    is_retryable,
)
from .models import (
    DEFAULT_MODELS,
    FALLBACK_MODELS,
    RegistryCheck,
    Role,
    resolve,
    short_name,
    validate_registry,
)
from .rerank import (
    actions_to_withdraw,
    rerank_lessons,
    retrieve_memories,
    split_by_polarity,
)
from .schemas import (
    ACTIONS,
    ALL_SCHEMAS,
    CRITIQUE_SCHEMA,
    DIAGNOSIS_SCHEMA,
    PROPOSAL_SCHEMA,
    SECOND_OPINION_SCHEMA,
    VISION_REPORT_SCHEMA,
)
from .scientist import choose_experiment, diagnose, propose_experiment, validate_proposal
from .structured import (
    complete_json,
    grammar_response_format,
    json_object_response_format,
    json_schema_response_format,
    parse_json,
    validate,
)
from .telemetry import CallRecord, UsageLedger
from .tools import (
    ActionHistory,
    DEFAULT_TRAINING_CONFIG,
    TOOL_SPECS,
    ToolCall,
    apply_experiment,
    apply_tool,
    diff_configs,
    force_tool,
    parse_tool_calls,
    tool_definitions,
)
from .vision import (
    collect_worst_crops,
    describe_dataset_sample,
    encode_image,
    inspect_misclassified,
)

__version__ = "0.1.0"

__all__ = [
    # client + config
    "FireworksClient",
    "FireworksSettings",
    "DEFAULT_BASE_URL",
    "build_chat_body",
    # roles + registry
    "Role",
    "DEFAULT_MODELS",
    "FALLBACK_MODELS",
    "RegistryCheck",
    "resolve",
    "short_name",
    "validate_registry",
    # the three frozen interfaces from the team plan
    "diagnose",
    "propose_experiment",
    "critique_result",
    # scientist / critic extras
    "choose_experiment",
    "validate_proposal",
    "to_lesson_document",
    "embedding_text",
    # tools
    "ACTIONS",
    "TOOL_SPECS",
    "ToolCall",
    "ActionHistory",
    "DEFAULT_TRAINING_CONFIG",
    "tool_definitions",
    "force_tool",
    "parse_tool_calls",
    "apply_tool",
    "apply_experiment",
    "diff_configs",
    # structured output
    "complete_json",
    "json_schema_response_format",
    "json_object_response_format",
    "grammar_response_format",
    "parse_json",
    "validate",
    "ALL_SCHEMAS",
    "DIAGNOSIS_SCHEMA",
    "PROPOSAL_SCHEMA",
    "CRITIQUE_SCHEMA",
    "VISION_REPORT_SCHEMA",
    "SECOND_OPINION_SCHEMA",
    # memory
    "FireworksEmbedder",
    "LangChainFireworksEmbeddings",
    "attach_embeddings",
    "cosine_similarity",
    "rerank_lessons",
    "retrieve_memories",
    "split_by_polarity",
    "actions_to_withdraw",
    # vision
    "inspect_misclassified",
    "describe_dataset_sample",
    "collect_worst_crops",
    "encode_image",
    # telemetry
    "UsageLedger",
    "CallRecord",
    # errors
    "FireworksError",
    "FireworksConfigError",
    "FireworksAuthError",
    "FireworksNotFoundError",
    "FireworksRateLimitError",
    "FireworksServerError",
    "FireworksTimeoutError",
    "FireworksBadRequestError",
    "FireworksStructuredOutputError",
    "FireworksToolChoiceError",
    "is_retryable",
]
