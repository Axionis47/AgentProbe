"""Singleton wrapper around the Chroma HTTP client.

`chromadb.HttpClient` is a factory function (not a type), so we keep the
returned client untyped and rely on Chroma's runtime contract. The single
collection in use today is `conversations`; it stores one embedding per
completed conversation, with metadata to enable filtered similarity search.
"""

from __future__ import annotations

from typing import Any

import chromadb

from app.config import settings


class ChromaDBClient:
    _client: Any = None

    @classmethod
    def get_client(cls) -> Any:
        if cls._client is None:
            cls._client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port,
            )
        return cls._client

    @classmethod
    def get_conversations_collection(cls) -> Any:
        client = cls.get_client()
        return client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def reset(cls) -> None:
        """Clear the cached client — used by tests that swap fakes."""
        cls._client = None
