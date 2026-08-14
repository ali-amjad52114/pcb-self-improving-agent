"""LangChain / LangGraph wiring for Fireworks.

Person 3 owns the graph and should not have to care how this package talks
HTTP. These factories hand back objects the LangChain stack already knows how
to use, configured from the same `FireworksSettings` as everything else, so a
model override in `.env` applies to the graph too.

Two routes are supported:

* `chat_model()` returns `langchain_fireworks.ChatFireworks` when that package
  is installed. This is the route to use if the graph wants
  `.bind_tools()` / `.with_structured_output()`.
* `chat_model_via_openai()` returns `langchain_openai.ChatOpenAI` pointed at
  the Fireworks base URL. Fireworks is OpenAI-compatible, and `langchain-openai`
  is already in requirements.txt, so this works with zero new dependencies.

Neither is required for the agent itself: `scientist.py` and `critic.py` use
`FireworksClient` directly so that structured output, tool validation, retries,
and token accounting behave identically whether or not LangChain is present.
"""

from __future__ import annotations

from typing import Any

from .config import FireworksSettings
from .errors import FireworksConfigError
from .models import Role


def chat_model(
    role: Role = Role.SCIENTIST,
    settings: FireworksSettings | None = None,
    **kwargs: Any,
) -> Any:
    """`langchain_fireworks.ChatFireworks` for the given role."""
    try:
        from langchain_fireworks import ChatFireworks  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FireworksConfigError(
            "langchain-fireworks is not installed. Run "
            "`pip install langchain-fireworks`, or use chat_model_via_openai(), "
            "which needs only langchain-openai."
        ) from exc

    settings = settings or FireworksSettings.from_env()
    params: dict[str, Any] = {
        "model": settings.model_for(role),
        "api_key": settings.api_key,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    params.update(kwargs)
    return ChatFireworks(**params)


def chat_model_via_openai(
    role: Role = Role.SCIENTIST,
    settings: FireworksSettings | None = None,
    **kwargs: Any,
) -> Any:
    """`langchain_openai.ChatOpenAI` pointed at Fireworks.

    Fireworks-only parameters (`reasoning_effort`, `perf_metrics_in_response`,
    `prompt_cache_isolation_key`) are passed through `model_kwargs`, since the
    OpenAI client forwards unknown fields in the request body.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FireworksConfigError(
            "langchain-openai is not installed; it is listed in requirements.txt"
        ) from exc

    settings = settings or FireworksSettings.from_env()
    model_kwargs: dict[str, Any] = dict(kwargs.pop("model_kwargs", {}))
    if settings.reasoning_effort:
        model_kwargs.setdefault("reasoning_effort", settings.reasoning_effort)
    if settings.perf_metrics:
        model_kwargs.setdefault("perf_metrics_in_response", True)
    if settings.prompt_cache_isolation_key:
        model_kwargs.setdefault(
            "prompt_cache_isolation_key", settings.prompt_cache_isolation_key
        )

    params: dict[str, Any] = {
        "model": settings.model_for(role),
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "timeout": settings.timeout,
        "max_retries": settings.max_retries,
    }
    if model_kwargs:
        params["model_kwargs"] = model_kwargs
    params.update(kwargs)
    return ChatOpenAI(**params)


def embeddings(settings: FireworksSettings | None = None, **kwargs: Any) -> Any:
    """LangChain-compatible embeddings backed by this package's client.

    Returns the local adapter rather than `langchain_fireworks.FireworksEmbeddings`
    so batching, the request cache, and the usage ledger still apply. Drop it
    straight into `MongoDBAtlasVectorSearch(embedding=...)`.
    """
    from .client import FireworksClient
    from .embeddings import FireworksEmbedder, LangChainFireworksEmbeddings

    client = kwargs.pop("client", None) or FireworksClient(settings)
    return LangChainFireworksEmbeddings(FireworksEmbedder(client, **kwargs))


def bind_experiment_tools(model: Any, allowed: list[str] | None = None) -> Any:
    """Attach the experiment allowlist to a LangChain chat model.

    Only useful if the graph calls the model itself. `scientist.propose_experiment`
    already does this, with argument validation the LangChain path does not
    perform.
    """
    from .tools import tool_definitions

    if not hasattr(model, "bind_tools"):
        raise FireworksConfigError(
            f"{type(model).__name__} has no bind_tools(); use a chat model that "
            "supports tool calling"
        )
    return model.bind_tools(tool_definitions(allowed), tool_choice="required")


def langsmith_metadata(settings: FireworksSettings | None = None) -> dict[str, Any]:
    """Run metadata for LangSmith traces.

    The hackathon ships LangSmith credit, and tagging traces with the model
    roster is what lets you tell a Run A trace from a Run B trace afterwards.
    """
    settings = settings or FireworksSettings.from_env(require_key=False)
    return {
        "provider": "fireworks",
        "fireworks_models": {
            role.value: mid for role, mid in sorted(
                settings.models.items(), key=lambda kv: kv[0].value
            )
        },
        "fireworks_base_url": settings.base_url,
    }
