"""Agent and User persona definitions.

Personas are pure data — no logic. Loaded from DB models (AgentConfig, Scenario).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import settings


@dataclass
class AgentPersona:
    """Configuration for the agent under test. Loaded from agent_configs table."""

    name: str
    system_prompt: str
    model: str
    temperature: float = settings.default_temperature
    max_tokens: int = settings.default_max_tokens
    tools: list[dict[str, Any]] = field(default_factory=list)
    agent_type: str = "builtin"
    endpoint_url: str | None = None

    @classmethod
    def from_db(cls, agent_config: Any) -> AgentPersona:
        return cls(
            name=agent_config.name,
            system_prompt=agent_config.system_prompt,
            model=agent_config.model,
            temperature=agent_config.temperature,
            max_tokens=agent_config.max_tokens,
            tools=agent_config.tools or [],
            agent_type=getattr(agent_config, 'agent_type', 'builtin'),
            endpoint_url=getattr(agent_config, 'endpoint_url', None),
        )


@dataclass
class UserPersona:
    """Configuration for the simulated user. Loaded from scenario.user_persona."""

    personality: str = "neutral"
    expertise_level: str = "intermediate"
    goal: str = "Get help with a task"
    model: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            self.model = settings.user_simulator_model

    @property
    def system_prompt(self) -> str:
        return f"""You are simulating a real user in a conversation with an AI assistant.

Your persona:
- Personality: {self.personality}
- Expertise level: {self.expertise_level}
- Goal: {self.goal}

Guidelines:
- Stay in character throughout the entire conversation
- React naturally to the assistant's responses
- If the assistant solves your problem, say [GOAL_ACHIEVED] in your message
- If the assistant is unhelpful after 3+ turns, say [FRUSTRATED] in your message
- Keep responses concise (1-3 sentences typically)
- Ask follow-up questions if the answer is unclear
- Do NOT break character or acknowledge you are simulating"""

    @classmethod
    def from_dict(cls, data: dict[str, Any], model: str = "") -> UserPersona:
        return cls(
            personality=data.get("personality", "neutral"),
            expertise_level=data.get("expertise_level", "intermediate"),
            goal=data.get("goal", "Get help with a task"),
            model=model or data.get("model", settings.user_simulator_model),
        )
