"""Unit tests for ToolSimulator."""

from __future__ import annotations

import json

import pytest

from app.engine.environment import SimulationEnvironment
from app.engine.tool_simulator import ToolSimulator
from app.engine.types import ToolCall


@pytest.fixture
def env_normal() -> SimulationEnvironment:
    return SimulationEnvironment(tool_failure_rate=0.0)


@pytest.fixture
def env_failing() -> SimulationEnvironment:
    return SimulationEnvironment(tool_failure_rate=1.0)


@pytest.fixture
def env_degraded() -> SimulationEnvironment:
    return SimulationEnvironment(tool_failure_rate=0.0, tool_latency_ms=100)


class TestToolSimulatorNormal:
    """Normal mode — tools return expected output."""

    @pytest.mark.asyncio
    async def test_known_tool_returns_response(self, env_normal: SimulationEnvironment) -> None:
        sim = ToolSimulator(environment=env_normal)
        call = ToolCall(id="call_1", name="get_weather", arguments={"city": "London"})

        result = await sim.execute(call)

        assert not result.is_error
        data = json.loads(result.content)
        assert "temperature" in data

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_acknowledgment(self, env_normal: SimulationEnvironment) -> None:
        sim = ToolSimulator(environment=env_normal)
        call = ToolCall(id="call_2", name="custom_tool", arguments={"foo": "bar"})

        result = await sim.execute(call)

        assert not result.is_error
        data = json.loads(result.content)
        assert data["status"] == "success"
        assert data["tool"] == "custom_tool"

    @pytest.mark.asyncio
    async def test_partial_name_match(self, env_normal: SimulationEnvironment) -> None:
        sim = ToolSimulator(environment=env_normal)
        call = ToolCall(id="call_3", name="search_documents", arguments={"query": "test"})

        result = await sim.execute(call)

        assert not result.is_error
        data = json.loads(result.content)
        assert "results" in data

    @pytest.mark.asyncio
    async def test_custom_responses(self, env_normal: SimulationEnvironment) -> None:
        custom = {"my_tool": json.dumps({"custom": True})}
        sim = ToolSimulator(environment=env_normal, custom_responses=custom)
        call = ToolCall(id="call_4", name="my_tool", arguments={})

        result = await sim.execute(call)

        data = json.loads(result.content)
        assert data["custom"] is True


class TestToolSimulatorFailures:
    """Failing mode — tools return errors based on failure rate."""

    @pytest.mark.asyncio
    async def test_100_percent_failure_rate(self, env_failing: SimulationEnvironment) -> None:
        sim = ToolSimulator(environment=env_failing)
        call = ToolCall(id="call_5", name="search", arguments={"query": "test"})

        result = await sim.execute(call)

        assert result.is_error
        data = json.loads(result.content)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_zero_failure_rate_never_fails(self, env_normal: SimulationEnvironment) -> None:
        sim = ToolSimulator(environment=env_normal)
        call = ToolCall(id="call_6", name="search", arguments={"query": "test"})

        # Run 20 times — should never fail
        for _ in range(20):
            result = await sim.execute(call)
            assert not result.is_error


class TestToolSimulatorDegraded:
    """Degraded mode — tools add latency."""

    @pytest.mark.asyncio
    async def test_latency_injection(self, env_degraded: SimulationEnvironment) -> None:
        import time

        sim = ToolSimulator(environment=env_degraded)
        call = ToolCall(id="call_7", name="search", arguments={"query": "test"})

        start = time.perf_counter()
        result = await sim.execute(call)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert not result.is_error
        assert elapsed_ms >= 90  # ~100ms with some tolerance
