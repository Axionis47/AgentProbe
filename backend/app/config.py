import os

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "AgentProbe"
    debug: bool = False
    api_key: str = "changeme"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "agentprobe"
    postgres_password: str = "agentprobe"
    postgres_db: str = "agentprobe"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "agentprobe-consumers"

    # ChromaDB
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000

    # LLM Provider (model-agnostic via LiteLLM)
    # Provider: "vertex_ai", "ollama", "anthropic", "openai"
    llm_provider: str = "vertex_ai"
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    # API keys (optional — only needed for cloud providers)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Vertex AI settings (uses Application Default Credentials)
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    # Model names (LiteLLM format: "provider/model" or just "model" for Ollama)
    default_model: str = "vertex_ai/gemini-2.0-flash"
    judge_model: str = "vertex_ai/gemini-2.0-flash"
    user_simulator_model: str = "vertex_ai/gemini-2.0-flash"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    model_config = {"env_prefix": "AGENTPROBE_", "env_file": ".env"}

    @model_validator(mode="after")
    def _fill_vertex_from_gcp_env(self) -> "Settings":
        """Fall back to standard GCP env vars for Vertex AI config."""
        if not self.vertex_project:
            self.vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT", "plotpointe")
        if not self.vertex_location or self.vertex_location == "us-central1":
            loc = os.getenv("GOOGLE_CLOUD_LOCATION", "")
            if loc:
                self.vertex_location = loc
        return self


settings = Settings()
