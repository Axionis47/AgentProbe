"""Tests for the deterministic FakeLLMClient and the make_llm_client factory.

The fake exists so tests and CI can run without a real LLM API. These tests
pin the behaviour the e2e test depends on:
- chat() returns *something* on every call
- the same input produces the same embedding (so similarity search is
  deterministic in the e2e test)
- the factory swaps based on llm_provider
"""

from __future__ import annotations

import pytest

from app.engine.llm_client import FakeLLMClient, LLMClient, make_llm_client
from app.engine.types import LLMResponse


@pytest.mark.asyncio
async def test_chat_returns_acknowledgement_when_no_tools():
    fake = FakeLLMClient()
    out = await fake.chat(
        model="anything",
        messages=[{"role": "user", "content": "hi there"}],
    )
    assert isinstance(out, LLMResponse)
    assert "hi there" in out.content
    assert out.tool_calls == []
    assert out.model == "fake"
    assert out.input_tokens > 0
    assert out.output_tokens > 0


@pytest.mark.asyncio
async def test_chat_emits_tool_call_on_first_turn_when_tools_provided():
    fake = FakeLLMClient()
    out = await fake.chat(
        model="anything",
        messages=[{"role": "user", "content": "look this up"}],
        tools=[{"name": "lookup_order", "description": "look up an order"}],
    )
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "lookup_order"
    assert out.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_chat_does_not_loop_tool_calls_forever():
    """Subsequent turns should fall back to plain text — otherwise the simulator hangs."""
    fake = FakeLLMClient()
    first = await fake.chat(
        model="x",
        messages=[{"role": "user", "content": "go"}],
        tools=[{"name": "t1"}],
    )
    second = await fake.chat(
        model="x",
        messages=[{"role": "user", "content": "again"}],
        tools=[{"name": "t1"}],
    )
    assert first.tool_calls != []
    assert second.tool_calls == []


@pytest.mark.asyncio
async def test_embed_is_deterministic_for_same_input():
    fake = FakeLLMClient()
    a = await fake.embed("hello world")
    b = await fake.embed("hello world")
    assert a == b


@pytest.mark.asyncio
async def test_embed_differs_for_different_inputs():
    fake = FakeLLMClient()
    a = await fake.embed("hello world")
    b = await fake.embed("goodbye world")
    assert a != b


@pytest.mark.asyncio
async def test_embed_returns_fixed_dimension():
    fake = FakeLLMClient()
    out = await fake.embed("anything")
    assert len(out) == 16
    assert all(isinstance(v, float) for v in out)


def test_make_llm_client_returns_real_client_by_default(monkeypatch):
    from app.engine import llm_client as mod

    monkeypatch.setattr(mod.settings, "llm_provider", "vertex_ai")
    assert isinstance(make_llm_client(), LLMClient)


def test_make_llm_client_returns_fake_when_provider_is_fake(monkeypatch):
    from app.engine import llm_client as mod

    monkeypatch.setattr(mod.settings, "llm_provider", "fake")
    assert isinstance(make_llm_client(), FakeLLMClient)
