"""Model-agnostic LLM client using LiteLLM.

Supports Ollama, Claude, OpenAI, and any LiteLLM-compatible provider.
Swap providers by changing one config value — zero code changes.

This is the ONLY file that talks to LLM APIs. Mock this for tests.
"""

from __future__ import annotations

import os
from typing import Any

import structlog
from litellm import acompletion

from app.config import settings
from app.engine.types import LLMResponse, ToolCall

logger = structlog.get_logger()


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
            model: LiteLLM model string (e.g., "ollama/mistral:7b-instruct",
                   "claude-sonnet-4-20250514", "gpt-4o")
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

        # Set Ollama API base for ollama models
        if model.startswith("ollama/"):
            kwargs["api_base"] = settings.ollama_base_url

        logger.debug(
            "llm_request",
            model=model,
            message_count=len(full_messages),
            has_tools=bool(tools),
        )

        response = await acompletion(**kwargs)

        # Normalize response
        message = response.choices[0].message
        content = message.content or ""

        # Extract tool calls if present
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                # LiteLLM may return args as string or dict
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}

                tool_calls.append(
                    ToolCall(
                        id=tc.id or f"call_{id(tc)}",
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # Extract usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

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
