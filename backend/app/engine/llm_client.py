"""Model-agnostic LLM client using LiteLLM.

Supports Vertex AI (Gemini), Ollama, Claude, OpenAI, and any LiteLLM-compatible
provider. Swap providers by changing one config value — zero code changes.

This is the ONLY file that talks to LLM APIs. Mock this for tests.
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog
from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)

from app.config import settings
from app.engine.types import LLMResponse, ToolCall

logger = structlog.get_logger()

# Vertex AI specific error messages for better diagnostics
_VERTEX_HINTS: dict[str, str] = {
    "Could not automatically determine credentials": (
        "Vertex AI ADC not configured. Run: gcloud auth application-default login"
    ),
    "403": "Vertex AI access denied. Check IAM permissions for the service account.",
    "404": "Model or endpoint not found. Verify model name and region.",
    "429": "Vertex AI quota exceeded. Check quotas in GCP console.",
}


class LLMClient:
    """Unified LLM client wrapping LiteLLM."""

    def __init__(self) -> None:
        # Set API keys if configured
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        # Set Ollama base URL
        if settings.llm_provider == "ollama":
            os.environ["OLLAMA_API_BASE"] = settings.ollama_base_url

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request to any LLM provider.

        Args:
            model: LiteLLM model string (e.g., "vertex_ai/gemini-2.0-flash",
                   "ollama/mistral:7b-instruct", "claude-sonnet-4-20250514", "gpt-4o")
            messages: Chat messages in OpenAI format
            system: System prompt (prepended as system message)
            tools: Tool definitions in OpenAI function calling format
            temperature: Sampling temperature
            max_tokens: Maximum output tokens

        Returns:
            Normalized LLMResponse regardless of provider
        """
        # Build messages with optional system prompt
        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools

        # Provider-specific config
        if model.startswith("ollama/"):
            kwargs["api_base"] = settings.ollama_base_url
        elif model.startswith("vertex_ai/"):
            kwargs["vertex_project"] = settings.vertex_project
            kwargs["vertex_location"] = settings.vertex_location

        logger.debug(
            "llm_request",
            model=model,
            message_count=len(full_messages),
            has_tools=bool(tools),
        )

        try:
            response = await acompletion(**kwargs)
        except RateLimitError as exc:
            logger.error("llm_rate_limit", model=model, error=str(exc))
            raise
        except AuthenticationError as exc:
            _log_vertex_hint(str(exc))
            logger.error("llm_auth_error", model=model, error=str(exc))
            raise
        except NotFoundError as exc:
            _log_vertex_hint(str(exc))
            logger.error("llm_not_found", model=model, error=str(exc))
            raise
        except APIConnectionError as exc:
            logger.error("llm_connection_error", model=model, error=str(exc))
            raise
        except APIError as exc:
            _log_vertex_hint(str(exc))
            logger.error("llm_api_error", model=model, error=str(exc))
            raise

        # Normalize response
        message = response.choices[0].message
        content = message.content or ""

        # Extract tool calls if present
        tool_calls = _extract_tool_calls(message)

        # Extract usage — normalize across providers
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0

        result = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=response.model or model,
            stop_reason=response.choices[0].finish_reason or "",
        )

        logger.debug(
            "llm_response",
            model=model,
            content_length=len(content),
            tool_call_count=len(tool_calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return result


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    """Extract and normalize tool calls from any provider's response.

    Handles differences in how providers (OpenAI, Gemini, Claude) return
    function/tool calls. LiteLLM normalizes most of this, but edge cases
    remain — especially around argument serialization and missing IDs.
    """
    raw_calls = getattr(message, "tool_calls", None)
    if not raw_calls:
        return []

    tool_calls: list[ToolCall] = []
    for idx, tc in enumerate(raw_calls):
        fn = getattr(tc, "function", None)
        if fn is None:
            continue

        name = getattr(fn, "name", None) or ""
        args = getattr(fn, "arguments", None)

        # LiteLLM may return args as string or dict
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        elif args is None:
            args = {}

        # Gemini sometimes omits tool_call IDs; generate a stable fallback
        call_id = getattr(tc, "id", None) or f"call_{idx}"

        tool_calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=args,
            )
        )
    return tool_calls


def _log_vertex_hint(error_msg: str) -> None:
    """Log a helpful hint if the error matches a known Vertex AI pattern."""
    for pattern, hint in _VERTEX_HINTS.items():
        if pattern in error_msg:
            logger.warning("vertex_ai_hint", hint=hint)
            return
