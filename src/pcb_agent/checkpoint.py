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
            from langgraph.checkpoint.mongodb import MongoDBSaver
            from pcb_memory.client import get_client

            # Reuse Person 1's process-wide pooled client rather than opening a
            # second Atlas connection for checkpoints.
            client = get_client()
            # Fail clearly if the cluster is unreachable.
            client.admin.command("ping")
            return MongoDBSaver(
                client=client,
                db_name=settings.mongodb_db_name,
                checkpoint_collection_name="langgraph_checkpoints",
                writes_collection_name="langgraph_checkpoint_writes",
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
