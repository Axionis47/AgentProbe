"""Tests for the model-agnostic LLM client.

All tests mock litellm.acompletion — no real API calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.engine.llm_client import LLMClient, _extract_tool_calls
from app.engine.types import ToolCall


# ---------------------------------------------------------------------------
# Helpers — build mock LiteLLM responses
# ---------------------------------------------------------------------------


def _make_response(
    content: str = "hello",
    tool_calls: list | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "vertex_ai/gemini-2.0-flash",
    finish_reason: str = "stop",
):
    """Return a SimpleNamespace that mimics a LiteLLM ModelResponse."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _make_tool_call(
    call_id: str = "call_1",
    name: str = "get_weather",
    arguments: str | dict = '{"city": "London"}',
):
    """Return a SimpleNamespace that mimics a tool call object."""
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, function=fn)


# ---------------------------------------------------------------------------
# Vertex AI provider configuration
# ---------------------------------------------------------------------------


class TestVertexAIConfig:
    """Verify that Vertex AI params are passed to litellm."""

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_vertex_params_passed(self, mock_completion):
        """vertex_project and vertex_location are forwarded for vertex_ai models."""
        mock_completion.return_value = _make_response()

        client = LLMClient()
        await client.chat(
            model="vertex_ai/gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
        )

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["vertex_project"]  # non-empty
        assert call_kwargs["vertex_location"]  # non-empty
        assert call_kwargs["model"] == "vertex_ai/gemini-2.0-flash"

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_ollama_no_vertex_params(self, mock_completion):
        """Ollama models should NOT receive vertex_* params."""
        mock_completion.return_value = _make_response(model="ollama/mistral:7b-instruct")

        client = LLMClient()
        await client.chat(
            model="ollama/mistral:7b-instruct",
            messages=[{"role": "user", "content": "hi"}],
        )

        call_kwargs = mock_completion.call_args.kwargs
        assert "vertex_project" not in call_kwargs
        assert "vertex_location" not in call_kwargs
        assert "api_base" in call_kwargs  # ollama gets api_base instead

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_openai_no_vertex_params(self, mock_completion):
        """OpenAI models should NOT receive vertex_* params."""
        mock_completion.return_value = _make_response(model="gpt-4o")

        client = LLMClient()
        await client.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

        call_kwargs = mock_completion.call_args.kwargs
        assert "vertex_project" not in call_kwargs
        assert "vertex_location" not in call_kwargs


# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------


class TestToolCallExtraction:
    """Test _extract_tool_calls with various provider response shapes."""

    def test_no_tool_calls(self):
        """Message with no tool_calls returns empty list."""
        msg = SimpleNamespace(content="just text", tool_calls=None)
        assert _extract_tool_calls(msg) == []

    def test_standard_tool_call_string_args(self):
        """Standard OpenAI-style tool call with JSON string arguments."""
        tc = _make_tool_call(call_id="call_abc", name="search", arguments='{"q": "test"}')
        msg = SimpleNamespace(content="", tool_calls=[tc])

        result = _extract_tool_calls(msg)
        assert len(result) == 1
        assert result[0].id == "call_abc"
        assert result[0].name == "search"
        assert result[0].arguments == {"q": "test"}

    def test_tool_call_dict_args(self):
        """Gemini sometimes returns arguments as a dict (already parsed)."""
        tc = _make_tool_call(call_id="call_1", name="lookup", arguments={"id": 42})
        msg = SimpleNamespace(content="", tool_calls=[tc])

        result = _extract_tool_calls(msg)
        assert len(result) == 1
        assert result[0].arguments == {"id": 42}

    def test_tool_call_missing_id(self):
        """Gemini may omit tool_call IDs; fallback should be generated."""
        tc = SimpleNamespace(
            id=None,
            function=SimpleNamespace(name="do_thing", arguments="{}"),
        )
        msg = SimpleNamespace(content="", tool_calls=[tc])

        result = _extract_tool_calls(msg)
        assert len(result) == 1
        assert result[0].id == "call_0"  # index-based fallback

    def test_tool_call_invalid_json_args(self):
        """Malformed JSON arguments wrapped in {raw: ...}."""
        tc = _make_tool_call(call_id="call_x", name="broken", arguments="not json at all")
        msg = SimpleNamespace(content="", tool_calls=[tc])

        result = _extract_tool_calls(msg)
        assert result[0].arguments == {"raw": "not json at all"}

    def test_tool_call_none_args(self):
        """None arguments normalized to empty dict."""
        tc = SimpleNamespace(
            id="call_n",
            function=SimpleNamespace(name="no_args", arguments=None),
        )
        msg = SimpleNamespace(content="", tool_calls=[tc])

        result = _extract_tool_calls(msg)
        assert result[0].arguments == {}

    def test_multiple_tool_calls(self):
        """Multiple tool calls extracted in order."""
        tcs = [
            _make_tool_call(call_id="c1", name="fn_a", arguments='{"x": 1}'),
            _make_tool_call(call_id="c2", name="fn_b", arguments='{"y": 2}'),
        ]
        msg = SimpleNamespace(content="", tool_calls=tcs)

        result = _extract_tool_calls(msg)
        assert len(result) == 2
        assert result[0].name == "fn_a"
        assert result[1].name == "fn_b"


# ---------------------------------------------------------------------------
# Token counting normalization
# ---------------------------------------------------------------------------


class TestTokenCounting:
    """Verify token counts are normalized from provider responses."""

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_standard_usage(self, mock_completion):
        mock_completion.return_value = _make_response(prompt_tokens=100, completion_tokens=50)

        client = LLMClient()
        result = await client.chat(
            model="vertex_ai/gemini-2.0-flash",
            messages=[{"role": "user", "content": "count tokens"}],
        )
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_missing_usage(self, mock_completion):
        """When usage is None, tokens default to 0."""
        resp = _make_response()
        resp.usage = None
        mock_completion.return_value = resp

        client = LLMClient()
        result = await client.chat(
            model="vertex_ai/gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_none_token_fields(self, mock_completion):
        """When individual token fields are None, they default to 0."""
        resp = _make_response()
        resp.usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None)
        mock_completion.return_value = resp

        client = LLMClient()
        result = await client.chat(
            model="vertex_ai/gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result.input_tokens == 0
        assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify Vertex AI and general LLM errors are handled properly."""

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_rate_limit_error(self, mock_completion):
        from litellm.exceptions import RateLimitError

        mock_completion.side_effect = RateLimitError(
            message="429 quota exceeded",
            llm_provider="vertex_ai",
            model="vertex_ai/gemini-2.0-flash",
        )

        client = LLMClient()
        with pytest.raises(RateLimitError):
            await client.chat(
                model="vertex_ai/gemini-2.0-flash",
                messages=[{"role": "user", "content": "hi"}],
            )

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_auth_error(self, mock_completion):
        from litellm.exceptions import AuthenticationError

        mock_completion.side_effect = AuthenticationError(
            message="Could not automatically determine credentials",
            llm_provider="vertex_ai",
            model="vertex_ai/gemini-2.0-flash",
        )

        client = LLMClient()
        with pytest.raises(AuthenticationError):
            await client.chat(
                model="vertex_ai/gemini-2.0-flash",
                messages=[{"role": "user", "content": "hi"}],
            )

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_not_found_error(self, mock_completion):
        from litellm.exceptions import NotFoundError

        mock_completion.side_effect = NotFoundError(
            message="404 Model not found",
            llm_provider="vertex_ai",
            model="vertex_ai/gemini-nonexistent",
        )

        client = LLMClient()
        with pytest.raises(NotFoundError):
            await client.chat(
                model="vertex_ai/gemini-nonexistent",
                messages=[{"role": "user", "content": "hi"}],
            )

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_system_prompt_prepended(self, mock_completion):
        """System prompt is prepended as first message."""
        mock_completion.return_value = _make_response()

        client = LLMClient()
        await client.chat(
            model="vertex_ai/gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
            system="You are a helpful assistant.",
        )

        call_kwargs = mock_completion.call_args.kwargs
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a helpful assistant."
        assert msgs[1]["role"] == "user"

    @patch("app.engine.llm_client.acompletion", new_callable=AsyncMock)
    async def test_tools_forwarded(self, mock_completion):
        """Tool definitions are forwarded to the completion call."""
        mock_completion.return_value = _make_response()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        client = LLMClient()
        await client.chat(
            model="vertex_ai/gemini-2.0-flash",
            messages=[{"role": "user", "content": "weather?"}],
            tools=tools,
        )

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["tools"] == tools
