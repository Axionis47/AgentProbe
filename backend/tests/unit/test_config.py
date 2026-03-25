"""Unit tests for application configuration.

Tests default values, environment variable overrides,
database URL construction, and LLM provider settings.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.config import Settings


class TestDefaultValues:
    """Verify default configuration values load correctly."""

    def test_app_name_default(self):
        """Default app_name is AgentProbe."""
        s = Settings()
        assert s.app_name == "AgentProbe"

    def test_debug_default_false(self):
        """Debug is False by default."""
        s = Settings()
        assert s.debug is False

    def test_postgres_defaults(self):
        """PostgreSQL defaults are localhost:5432/agentprobe."""
        s = Settings()
        assert s.postgres_host == "localhost"
        assert s.postgres_port == 5432
        assert s.postgres_user == "agentprobe"
        assert s.postgres_password == "agentprobe"
        assert s.postgres_db == "agentprobe"

    def test_redis_url_default(self):
        """Default Redis URL."""
        s = Settings()
        assert s.redis_url == "redis://localhost:6379/0"

    def test_kafka_defaults(self):
        """Default Kafka settings."""
        s = Settings()
        assert s.kafka_bootstrap_servers == "localhost:9092"
        assert s.kafka_consumer_group == "agentprobe-consumers"

    def test_llm_provider_default(self):
        """Default LLM provider is vertex_ai."""
        s = Settings()
        assert s.llm_provider == "vertex_ai"

    def test_default_model(self):
        """Default model uses vertex_ai/gemini format."""
        s = Settings()
        assert "vertex_ai/" in s.default_model

    def test_celery_defaults(self):
        """Default Celery broker and result backend."""
        s = Settings()
        assert "redis://" in s.celery_broker_url
        assert "redis://" in s.celery_result_backend

    def test_api_key_default(self):
        """Default API key is the placeholder."""
        s = Settings()
        assert s.api_key == "changeme"


class TestDatabaseUrlConstruction:
    """Verify database_url and database_url_sync properties."""

    def test_async_database_url_format(self):
        """Async database URL uses postgresql+asyncpg driver."""
        s = Settings()
        url = s.database_url
        assert url.startswith("postgresql+asyncpg://")
        assert "agentprobe" in url
        assert "@localhost:5432/" in url

    def test_sync_database_url_format(self):
        """Sync database URL uses plain postgresql driver."""
        s = Settings()
        url = s.database_url_sync
        assert url.startswith("postgresql://")
        assert "asyncpg" not in url
        assert "@localhost:5432/" in url

    def test_database_url_includes_credentials(self):
        """Database URL includes user and password."""
        s = Settings()
        url = s.database_url
        assert "agentprobe:agentprobe@" in url

    def test_custom_postgres_settings_in_url(self):
        """Custom postgres settings are reflected in the URL."""
        s = Settings(
            postgres_host="db.example.com",
            postgres_port=5433,
            postgres_user="custom_user",
            postgres_password="secret",
            postgres_db="my_db",
        )
        url = s.database_url
        assert "custom_user:secret@db.example.com:5433/my_db" in url


class TestEnvironmentVariableOverrides:
    """Verify that AGENTPROBE_ prefixed env vars override defaults."""

    def test_env_prefix(self):
        """Settings use AGENTPROBE_ env prefix."""
        assert Settings.model_config["env_prefix"] == "AGENTPROBE_"

    def test_debug_override_via_env(self):
        """AGENTPROBE_DEBUG=true overrides debug to True."""
        with patch.dict(os.environ, {"AGENTPROBE_DEBUG": "true"}):
            s = Settings()
            assert s.debug is True

    def test_postgres_host_override_via_env(self):
        """AGENTPROBE_POSTGRES_HOST overrides postgres_host."""
        with patch.dict(os.environ, {"AGENTPROBE_POSTGRES_HOST": "remote-db"}):
            s = Settings()
            assert s.postgres_host == "remote-db"

    def test_postgres_port_override_via_env(self):
        """AGENTPROBE_POSTGRES_PORT overrides postgres_port."""
        with patch.dict(os.environ, {"AGENTPROBE_POSTGRES_PORT": "5433"}):
            s = Settings()
            assert s.postgres_port == 5433

    def test_llm_provider_override_via_env(self):
        """AGENTPROBE_LLM_PROVIDER overrides llm_provider."""
        with patch.dict(os.environ, {"AGENTPROBE_LLM_PROVIDER": "anthropic"}):
            s = Settings()
            assert s.llm_provider == "anthropic"

    def test_anthropic_api_key_override(self):
        """AGENTPROBE_ANTHROPIC_API_KEY overrides anthropic_api_key."""
        with patch.dict(os.environ, {"AGENTPROBE_ANTHROPIC_API_KEY": "sk-test-123"}):
            s = Settings()
            assert s.anthropic_api_key == "sk-test-123"

    def test_kafka_bootstrap_servers_override(self):
        """AGENTPROBE_KAFKA_BOOTSTRAP_SERVERS overrides kafka settings."""
        with patch.dict(os.environ, {"AGENTPROBE_KAFKA_BOOTSTRAP_SERVERS": "kafka:29092"}):
            s = Settings()
            assert s.kafka_bootstrap_servers == "kafka:29092"


class TestLLMProviderConfiguration:
    """Verify LLM provider related configuration fields."""

    def test_ollama_base_url_default(self):
        """Default Ollama base URL."""
        s = Settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_empty_api_keys_by_default(self):
        """API keys are empty strings by default (for local Ollama usage)."""
        s = Settings()
        assert s.anthropic_api_key == ""
        assert s.openai_api_key == ""

    def test_judge_model_default(self):
        """Judge model has a default value."""
        s = Settings()
        assert s.judge_model != ""

    def test_user_simulator_model_default(self):
        """User simulator model has a default value."""
        s = Settings()
        assert s.user_simulator_model != ""
