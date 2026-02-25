"""LLM-powered user simulator.

Generates realistic user messages based on a persona definition.
Uses a cheap model (e.g., Llama3 via Ollama) to keep costs low.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.engine.persona import UserPersona
from app.engine.types import LLMClientProtocol, Turn

logger = structlog.get_logger()


class UserSimulator:
    """Generates simulated user messages using an LLM."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        persona: UserPersona,
        initial_message: str = "",
    ) -> None:
        self.llm_client = llm_client
        self.persona = persona
        self.initial_message = initial_message

    async def generate(
        self,
        conversation_history: list[Turn],
        turn_index: int,
    ) -> str:
        """Generate the next user message based on conversation history.

        For turn 0, uses the initial message template if provided.
        For subsequent turns, asks the LLM to continue as the user persona.
        """
        # First turn: use template or generate from goal
        if turn_index == 0 and self.initial_message:
            logger.debug("user_simulator_using_template", turn=turn_index)
            return self.initial_message

        # Build conversation for the user sim LLM
        messages: list[dict[str, Any]] = []
        for turn in conversation_history:
            if turn.role == "user":
                # The user sim sees its own previous messages as "assistant"
                # (because the user sim IS generating user messages)
                messages.append({"role": "assistant", "content": turn.content})
            elif turn.role == "assistant":
                # The user sim sees the agent's messages as "user" input
                messages.append({"role": "user", "content": turn.content})

        # If no history yet, prompt the user sim to start the conversation
        if not messages:
            messages.append({
                "role": "user",
                "content": f"Start a conversation. Your goal: {self.persona.goal}",
            })

        response = await self.llm_client.chat(
            model=self.persona.model,
            messages=messages,
            system=self.persona.system_prompt,
            temperature=0.8,
            max_tokens=500,
        )

        logger.debug(
            "user_simulator_generated",
            turn=turn_index,
            content_length=len(response.content),
        )

        return response.content
