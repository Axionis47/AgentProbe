"""Tests for LLMClient.embed — the embedding wrapper used by similarity search.

Mocks litellm.aembedding so no real API calls happen.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.engine.llm_client import LLMClient


def _make_embedding_response(vector: list[float]) -> SimpleNamespace:
    """Mimic a LiteLLM EmbeddingResponse — `.data[0]` is a dict with 'embedding'."""
    return SimpleNamespace(data=[{"embedding": vector}])


@pytest.mark.asyncio
async def test_embed_returns_vector():
    client = LLMClient()
    expected = [0.1, 0.2, 0.3]

    with patch(
        "app.engine.llm_client.aembedding",
        AsyncMock(return_value=_make_embedding_response(expected)),
    ):
        got = await client.embed("hello world")

    assert got == expected


@pytest.mark.asyncio
async def test_embed_uses_configured_default_model():
    client = LLMClient()
    captured: dict = {}

    async def fake(**kwargs):
        captured.update(kwargs)
        return _make_embedding_response([0.0, 0.0])

    with patch("app.engine.llm_client.aembedding", side_effect=fake):
        await client.embed("x")

    assert captured["model"] == "vertex_ai/text-embedding-004"
    assert captured["input"] == ["x"]
    # Vertex provider needs project + location explicitly
    assert "vertex_project" in captured
    assert "vertex_location" in captured


@pytest.mark.asyncio
async def test_embed_respects_explicit_model_override():
    client = LLMClient()
    captured: dict = {}

    async def fake(**kwargs):
        captured.update(kwargs)
        return _make_embedding_response([0.5])

    with patch("app.engine.llm_client.aembedding", side_effect=fake):
        await client.embed("x", model="text-embedding-3-small")

    assert captured["model"] == "text-embedding-3-small"
    # Non-vertex model: should not pass vertex args
    assert "vertex_project" not in captured


@pytest.mark.asyncio
async def test_embed_handles_object_style_response():
    """LiteLLM occasionally returns objects rather than dicts in .data — handle both."""
    client = LLMClient()
    obj_response = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0])])

    with patch("app.engine.llm_client.aembedding", AsyncMock(return_value=obj_response)):
        got = await client.embed("anything")

    assert got == [1.0, 2.0]
