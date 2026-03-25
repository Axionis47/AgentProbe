"""Test fixtures for AgentProbe.

Shared fixtures for unit, integration, and e2e tests.
All external dependencies (DB, LLM, Redis, Kafka, Celery) are mocked.
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.db.session import get_db
from app.engine.environment import SimulationEnvironment
from app.engine.persona import AgentPersona, UserPersona
from app.engine.types import (
    ConversationResult,
    LLMResponse,
    ToolCall,
    ToolResult,
    Turn,
)
from app.evaluation.types import DEFAULT_DIMENSIONS, EvaluationResult, MetricValue, RubricDimension
from app.main import app
from app.models.base import Base

# Use a separate test database
TEST_DATABASE_URL = settings.database_url.replace("/agentprobe", "/agentprobe_test")


# ============================================================
# Event loop fixture
# ============================================================


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================
# Database fixtures
# ============================================================


@pytest.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ============================================================
# Mock database session (for unit tests without real DB)
# ============================================================


@pytest.fixture
def mock_db() -> AsyncMock:
    """Async mock database session that records add/flush/execute calls."""
    db = AsyncMock(spec=AsyncSession)
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock()
    return db


# ============================================================
# Mock LLM client
# ============================================================


def make_llm_response(
    content: str = "OK",
    tool_calls: list[ToolCall] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> LLMResponse:
    """Create a predictable LLM response for testing."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model="test-model",
        stop_reason="end_turn",
    )


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Mock LLM client that returns predictable responses."""
    client = AsyncMock()
    client.chat = AsyncMock(return_value=make_llm_response("Test response"))
    return client


# ============================================================
# Sample data factories
# ============================================================


def make_agent_config_data(**overrides: Any) -> dict[str, Any]:
    """Factory for AgentConfigCreate-compatible data."""
    data = {
        "name": "Test Agent",
        "description": "A test agent config",
        "system_prompt": "You are a helpful assistant.",
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 4096,
        "tools": [],
        "metadata": {},
    }
    data.update(overrides)
    return data


def make_scenario_data(**overrides: Any) -> dict[str, Any]:
    """Factory for ScenarioCreate-compatible data."""
    data = {
        "name": "Test Scenario",
        "description": "A test scenario",
        "category": "general",
        "turns_template": [{"role": "user", "content": "Hello, help me with Python"}],
        "user_persona": {"personality": "neutral", "expertise_level": "intermediate", "goal": "Get help"},
        "constraints": {"max_turns": 5, "max_total_tokens": 50000},
        "difficulty": "medium",
        "tags": ["test"],
    }
    data.update(overrides)
    return data


def make_rubric_data(**overrides: Any) -> dict[str, Any]:
    """Factory for RubricCreate-compatible data."""
    data = {
        "name": "Test Rubric",
        "description": "A test rubric",
        "dimensions": [
            {
                "name": "helpfulness",
                "description": "How helpful is the assistant",
                "weight": 0.5,
                "criteria": ["Addresses user needs"],
            },
            {
                "name": "accuracy",
                "description": "How accurate is the response",
                "weight": 0.5,
                "criteria": ["Factually correct"],
            },
        ],
    }
    data.update(overrides)
    return data


def make_eval_run_data(**overrides: Any) -> dict[str, Any]:
    """Factory for EvalRunCreate-compatible data."""
    data = {
        "agent_config_id": "00000000-0000-0000-0000-000000000001",
        "scenario_id": "00000000-0000-0000-0000-000000000002",
        "rubric_id": None,
        "num_conversations": 5,
        "config": {},
    }
    data.update(overrides)
    return data


def make_conversation_turns(n_turns: int = 3) -> list[dict[str, Any]]:
    """Factory for conversation turn data stored as JSONB."""
    turns = []
    for i in range(n_turns):
        turns.append({"role": "user", "content": f"User message {i}"})
        turns.append({
            "role": "assistant",
            "content": f"Assistant response {i}",
            "latency_ms": 100 + i * 10,
            "input_tokens": 10 + i,
            "output_tokens": 20 + i,
        })
    return turns


def make_mock_conversation(**overrides: Any) -> MagicMock:
    """Factory for mock Conversation DB objects."""
    conv = MagicMock()
    conv.id = overrides.get("id", "conv-test-123")
    conv.eval_run_id = overrides.get("eval_run_id", "run-test-123")
    conv.turns = overrides.get("turns", make_conversation_turns(2))
    conv.turn_count = overrides.get("turn_count", 2)
    conv.total_tokens = overrides.get("total_tokens", 60)
    conv.total_input_tokens = overrides.get("total_input_tokens", 20)
    conv.total_output_tokens = overrides.get("total_output_tokens", 40)
    conv.total_latency_ms = overrides.get("total_latency_ms", 200)
    conv.status = overrides.get("status", "completed")
    conv.sequence_num = overrides.get("sequence_num", 0)
    conv.error_message = overrides.get("error_message", None)
    conv.metadata_ = overrides.get("metadata_", {})
    return conv


def make_mock_eval_run(**overrides: Any) -> MagicMock:
    """Factory for mock EvalRun DB objects."""
    run = MagicMock()
    run.id = overrides.get("id", "run-test-123")
    run.name = overrides.get("name", "Test Run")
    run.agent_config_id = overrides.get("agent_config_id", "config-test-123")
    run.scenario_id = overrides.get("scenario_id", "scenario-test-123")
    run.rubric_id = overrides.get("rubric_id", None)
    run.status = overrides.get("status", "pending")
    run.num_conversations = overrides.get("num_conversations", 5)
    run.config = overrides.get("config", {})
    run.error_message = overrides.get("error_message", None)
    run.started_at = overrides.get("started_at", None)
    run.completed_at = overrides.get("completed_at", None)
    run.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    return run


def make_mock_agent_config(**overrides: Any) -> MagicMock:
    """Factory for mock AgentConfig DB objects."""
    config = MagicMock()
    config.id = overrides.get("id", "config-test-123")
    config.name = overrides.get("name", "Test Agent")
    config.description = overrides.get("description", "Test description")
    config.system_prompt = overrides.get("system_prompt", "You are a helpful assistant.")
    config.model = overrides.get("model", "test-model")
    config.temperature = overrides.get("temperature", 0.7)
    config.max_tokens = overrides.get("max_tokens", 4096)
    config.tools = overrides.get("tools", [])
    config.metadata_ = overrides.get("metadata_", {})
    config.is_active = overrides.get("is_active", True)
    config.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    config.updated_at = overrides.get("updated_at", datetime.now(timezone.utc))
    return config


def make_mock_scenario(**overrides: Any) -> MagicMock:
    """Factory for mock Scenario DB objects."""
    scenario = MagicMock()
    scenario.id = overrides.get("id", "scenario-test-123")
    scenario.name = overrides.get("name", "Test Scenario")
    scenario.description = overrides.get("description", "A test scenario")
    scenario.category = overrides.get("category", "general")
    scenario.turns_template = overrides.get(
        "turns_template",
        [{"role": "user", "content": "Hello, help me"}],
    )
    scenario.user_persona = overrides.get(
        "user_persona",
        {"personality": "neutral", "expertise_level": "intermediate", "goal": "Get help"},
    )
    scenario.constraints = overrides.get("constraints", {"max_turns": 5})
    scenario.difficulty = overrides.get("difficulty", "medium")
    scenario.tags = overrides.get("tags", ["test"])
    scenario.is_active = overrides.get("is_active", True)
    scenario.created_at = overrides.get("created_at", datetime.now(timezone.utc))
    scenario.updated_at = overrides.get("updated_at", datetime.now(timezone.utc))
    return scenario


# ============================================================
# Mock Celery app
# ============================================================


@pytest.fixture
def mock_celery_app():
    """Mock Celery application that captures task dispatches."""
    with patch("app.workers.simulation_tasks.run_simulation") as mock_task:
        mock_task.delay = MagicMock(return_value=MagicMock(id="task-123"))
        yield mock_task


# ============================================================
# Mock Kafka producer
# ============================================================


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer that captures produced events."""
    with patch("app.pipeline.producer.KafkaProducer") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance
