"""Tests for the external agent HTTP client.

Mocks httpx so we never make a real network call. We're verifying the
request shape (URL, JSON body), the response normalization (content +
tool_calls + zero token counts), defaults for missing fields, and
error propagation when the endpoint returns 4xx/5xx.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.engine.external_client import ExternalAgentClient
from app.engine.types import LLMResponse


def _fake_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Build a fake httpx Response with the given JSON payload."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data or {})

    if status_code >= 400:
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "boom", request=MagicMock(), response=resp
            )
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _patch_httpx_post(response: MagicMock):
    """Context-manager-style patch for httpx.AsyncClient — captures the call."""
    fake_client_instance = MagicMock()
    fake_client_instance.post = AsyncMock(return_value=response)
    fake_client_instance.__aenter__ = AsyncMock(return_value=fake_client_instance)
    fake_client_instance.__aexit__ = AsyncMock(return_value=False)

    return patch(
        "app.engine.external_client.httpx.AsyncClient",
        return_value=fake_client_instance,
    ), fake_client_instance


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_posts_messages_to_configured_endpoint():
    response = _fake_response(json_data={"content": "ok"})
    cm, client_instance = _patch_httpx_post(response)
    with cm:
        c = ExternalAgentClient(endpoint_url="https://agent.example.com/chat")
        await c.chat(model="ignored", messages=[{"role": "user", "content": "hi"}])

    client_instance.post.assert_awaited_once()
    args, kwargs = client_instance.post.call_args
    assert args[0] == "https://agent.example.com/chat"
    assert kwargs["json"] == {"messages": [{"role": "user", "content": "hi"}]}
    assert kwargs["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_chat_includes_optional_system_and_tools_in_body():
    response = _fake_response(json_data={"content": "ok"})
    cm, client_instance = _patch_httpx_post(response)
    with cm:
        c = ExternalAgentClient(endpoint_url="https://agent/")
        await c.chat(
            model="ignored",
            messages=[{"role": "user", "content": "hi"}],
            system="you are helpful",
            tools=[{"name": "lookup"}],
        )

    body = client_instance.post.call_args.kwargs["json"]
    assert body["system"] == "you are helpful"
    assert body["tools"] == [{"name": "lookup"}]


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_normalizes_response_with_content_only():
    response = _fake_response(json_data={"content": "the answer is 42"})
    cm, _ = _patch_httpx_post(response)
    with cm:
        c = ExternalAgentClient(endpoint_url="https://agent/")
        out = await c.chat(model="ignored", messages=[])

    assert isinstance(out, LLMResponse)
    assert out.content == "the answer is 42"
    assert out.tool_calls == []
    assert out.input_tokens == 0
    assert out.output_tokens == 0
    assert out.model == "external"
    assert out.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_chat_extracts_tool_calls_with_defaults_for_missing_id():
    payload = {
        "content": "",
        "tool_calls": [
            {"name": "search", "arguments": {"q": "weather"}},
            {"id": "call-x", "name": "lookup", "arguments": {}},
        ],
    }
    response = _fake_response(json_data=payload)
    cm, _ = _patch_httpx_post(response)
    with cm:
        c = ExternalAgentClient(endpoint_url="https://agent/")
        out = await c.chat(model="ignored", messages=[])

    assert len(out.tool_calls) == 2
    assert out.tool_calls[0].name == "search"
    assert out.tool_calls[0].id == "call_0"  # generated when missing
    assert out.tool_calls[0].arguments == {"q": "weather"}
    assert out.tool_calls[1].id == "call-x"  # preserved when present


@pytest.mark.asyncio
async def test_chat_handles_null_content_gracefully():
    """Some APIs return content=null when only tool_calls are present."""
    response = _fake_response(json_data={"content": None, "tool_calls": []})
    cm, _ = _patch_httpx_post(response)
    with cm:
        c = ExternalAgentClient(endpoint_url="https://agent/")
        out = await c.chat(model="ignored", messages=[])

    assert out.content == ""


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_raises_on_http_error():
    response = _fake_response(status_code=500, json_data={})
    cm, _ = _patch_httpx_post(response)
    with cm:
        c = ExternalAgentClient(endpoint_url="https://agent/")
        with pytest.raises(httpx.HTTPStatusError):
            await c.chat(model="ignored", messages=[])
