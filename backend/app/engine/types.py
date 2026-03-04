"""Core types and protocols for the simulation engine.

All engine components depend on these interfaces, not on concrete implementations.
This makes every component independently testable and swappable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# ============================================================
# Data Types
# ============================================================


@dataclass
class Turn:
    """A single turn in a conversation."""

    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolCall:
    """An agent's request to call a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from a tool execution (real or simulated)."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str = ""


@dataclass
class ConversationResult:
    """Complete result of a multi-turn simulation."""

    turns: list[Turn]
    turn_count: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: int
    status: str  # "completed" | "failed" | "goal_achieved" | "frustrated"
    error_message: str | None = None


# ============================================================
# Protocols (Interfaces) — mock these for tests
# ============================================================


class LLMClientProtocol(Protocol):
    """Interface for any LLM provider (Ollama, Claude, OpenAI)."""

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


class UserSimulatorProtocol(Protocol):
    """Interface for generating simulated user messages."""

    async def generate(
        self,
        conversation_history: list[Turn],
        turn_index: int,
    ) -> str: ...


class ToolSimulatorProtocol(Protocol):
    """Interface for simulating tool execution."""

    async def execute(self, tool_call: ToolCall) -> ToolResult: ...


class AdversarialStrategyProtocol(Protocol):
    """Interface for injecting adversarial behavior."""

    def should_inject(self, turn_index: int) -> bool: ...

    def generate_adversarial_input(self, turn_index: int) -> str: ...
