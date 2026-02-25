import chromadb

from app.config import settings


class ChromaDBClient:
    _client: chromadb.HttpClient | None = None

    @classmethod
    def get_client(cls) -> chromadb.HttpClient:
        if cls._client is None:
            cls._client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port,
            )
        return cls._client

    @classmethod
    def get_conversations_collection(cls) -> chromadb.Collection:
        client = cls.get_client()
        return client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"},
        )
