"""HTTP client for external agent endpoints.

Calls the user's chatbot API. The endpoint must accept:
  POST with JSON body: {"messages": [{"role": "user", "content": "..."}]}

And return JSON: {"content": "response text", "tool_calls": [...] or null}

Tool calls format (optional):
  [{"name": "tool_name", "arguments": {...}}]
"""

import httpx
import structlog
from app.engine.types import LLMResponse, ToolCall

logger = structlog.get_logger()


class ExternalAgentClient:
    """Calls an external chatbot API endpoint."""

    def __init__(self, endpoint_url: str, timeout: float = 30.0) -> None:
        self.endpoint_url = endpoint_url
        self.timeout = timeout

    async def chat(
        self,
        model: str,  # ignored for external agents
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send messages to the external endpoint and normalize the response."""

        payload = {"messages": messages}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        headers = {"Content-Type": "application/json"}

        logger.debug("external_agent_request", url=self.endpoint_url, message_count=len(messages))

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        # Normalize response
        content = data.get("content", "") or ""

        tool_calls = []
        raw_calls = data.get("tool_calls") or []
        for i, tc in enumerate(raw_calls):
            tool_calls.append(ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
            ))

        logger.debug("external_agent_response", content_length=len(content), tool_call_count=len(tool_calls))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=0,  # external agents don't report token usage
            output_tokens=0,
            model="external",
            stop_reason="end_turn",
        )
