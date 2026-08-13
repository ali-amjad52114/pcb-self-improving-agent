"""MongoDB vs in-memory LangGraph checkpointers.

Long-term experience memory (Person 1) is separate from this checkpoint state.
"""

from __future__ import annotations

from typing import Any

from pcb_agent.config import Settings
from pcb_agent.contracts import CheckpointError


def get_checkpointer(settings: Settings) -> Any:
    backend = (settings.checkpointer_backend or "memory").lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    if backend == "mongodb":
        if not settings.mongodb_uri:
            raise CheckpointError(
                "MongoDB checkpointing requested but unavailable.\n"
                "Set MONGODB_URI or explicitly switch CHECKPOINTER_BACKEND=memory."
            )
        try:
            from pymongo import MongoClient
            from langgraph.checkpoint.mongodb import MongoDBSaver

            client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Fail clearly if the cluster is unreachable.
            client.admin.command("ping")
            return MongoDBSaver(
                client=client,
                db_name=settings.mongodb_db_name,
            )
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(
                "MongoDB checkpointing requested but unavailable.\n"
                "Set MONGODB_URI or explicitly switch CHECKPOINTER_BACKEND=memory.\n"
                f"Underlying error: {exc}"
            ) from exc

    raise CheckpointError(
        f"Unknown CHECKPOINTER_BACKEND={settings.checkpointer_backend!r}. "
        "Use 'memory' or 'mongodb'."
    )
