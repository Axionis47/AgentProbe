"""Configurable tool call simulator.

Intercepts agent tool calls and returns simulated results.
Supports normal, degraded, failing, and adversarial modes.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import structlog

from app.engine.environment import SimulationEnvironment
from app.engine.types import ToolCall, ToolResult

logger = structlog.get_logger()


# Default tool responses for common tool types
DEFAULT_TOOL_RESPONSES: dict[str, str] = {
    "search": json.dumps({"results": [{"title": "Example Result", "snippet": "This is a simulated search result with relevant information."}]}),
    "get_weather": json.dumps({"temperature": 72, "condition": "sunny", "humidity": 45}),
    "run_code": json.dumps({"output": "Hello, World!", "exit_code": 0}),
    "read_file": json.dumps({"content": "# Example file content\nThis is simulated file data."}),
    "write_file": json.dumps({"status": "success", "bytes_written": 256}),
}


class ToolSimulator:
    """Simulates tool execution with configurable behavior.

    Modes (controlled by SimulationEnvironment):
    - Normal: returns expected output
    - Degraded: adds latency via tool_latency_ms
    - Failing: returns errors based on tool_failure_rate
    """

    def __init__(
        self,
        environment: SimulationEnvironment,
        custom_responses: dict[str, str] | None = None,
    ) -> None:
        self.env = environment
        self.responses = {**DEFAULT_TOOL_RESPONSES, **(custom_responses or {})}

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Simulate a tool call, applying environment conditions."""
        logger.debug(
            "tool_simulator_executing",
            tool_name=tool_call.name,
            tool_id=tool_call.id,
            failure_rate=self.env.tool_failure_rate,
        )

        # Simulate latency
        if self.env.tool_latency_ms > 0:
            await asyncio.sleep(self.env.tool_latency_ms / 1000.0)

        # Simulate failure based on failure rate
        if self.env.tool_failure_rate > 0 and random.random() < self.env.tool_failure_rate:
            logger.debug("tool_simulator_injected_failure", tool_name=tool_call.name)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({
                    "error": "Tool execution failed",
                    "message": f"Simulated failure for tool '{tool_call.name}'",
                }),
                is_error=True,
            )

        # Return simulated response
        response = self._get_response(tool_call)
        return ToolResult(
            tool_call_id=tool_call.id,
            content=response,
            is_error=False,
        )

    def _get_response(self, tool_call: ToolCall) -> str:
        """Get the simulated response for a tool call."""
        # Check for exact tool name match
        if tool_call.name in self.responses:
            return self.responses[tool_call.name]

        # Check for partial match (e.g., "search_docs" matches "search")
        for key, response in self.responses.items():
            if key in tool_call.name:
                return response

        # Default: return the arguments back as acknowledgment
        return json.dumps({
            "status": "success",
            "tool": tool_call.name,
            "input_received": tool_call.arguments,
            "message": f"Tool '{tool_call.name}' executed successfully.",
        })
