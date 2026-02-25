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
    # Provider: "ollama", "anthropic", "openai"
    llm_provider: str = "ollama"
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    # API keys (optional — only needed for cloud providers)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Model names (LiteLLM format: "provider/model" or just "model" for Ollama)
    default_model: str = "ollama/mistral:7b-instruct"
    judge_model: str = "ollama/mistral:7b-instruct"
    user_simulator_model: str = "ollama/llama3:8b-instruct-q4_K_M"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    model_config = {"env_prefix": "AGENTPROBE_", "env_file": ".env"}


settings = Settings()
