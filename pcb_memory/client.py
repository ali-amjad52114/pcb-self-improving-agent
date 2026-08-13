"""Single shared MongoClient for the long-running agent process.

One client for the whole process: pooling is on, TLS/auth cost is paid once.
Defaults are enough for a hackathon agent (low concurrency). We only set
timeouts so a hung Atlas call cannot stall LangGraph forever.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

load_dotenv()

DEFAULT_DB = "persistent_context"


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.environ.get("MONGODB_URI")
    if not uri or "<user>" in uri or "<cluster>" in uri:
        raise RuntimeError(
            "Set MONGODB_URI in .env to your Atlas Hackathon Sandbox connection string."
        )
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=8_000,
        connectTimeoutMS=8_000,
        socketTimeoutMS=20_000,
        retryWrites=True,
        appName="pcb-self-improving-agent",
    )


def get_db(name: str | None = None) -> Database:
    return get_client()[name or os.environ.get("MONGODB_DB", DEFAULT_DB)]
