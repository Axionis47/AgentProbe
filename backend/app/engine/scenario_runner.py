"""Core multi-turn conversation orchestrator.

This is the most complex file in the engine. It coordinates:
- User simulator generating messages
- Agent responding via LLM
- Tool calls being intercepted and simulated
- Adversarial injection at configured turns
- Environment constraint enforcement (max turns, tokens, timeout)
- Per-turn metrics recording

Depends on protocols, not implementations — fully testable with mocks.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from app.engine.adversarial import AdversarialStrategy, NoOpAdversarial
from app.engine.environment import SimulationEnvironment
from app.engine.persona import AgentPersona, UserPersona
from app.engine.tool_simulator import ToolSimulator
from app.engine.types import (
    ConversationResult,
    LLMClientProtocol,
    ToolCall,
    ToolResult,
    Turn,
)
from app.engine.user_simulator import UserSimulator

logger = structlog.get_logger()


class ScenarioRunner:
    """Executes a multi-turn conversation between an agent and simulated user.

    Flow per turn:
    1. Check adversarial injection → override user message if triggered
    2. User simulator generates next user message (or uses template)
    3. Agent generates response (possibly with tool calls)
    4. If tool calls present, ToolSimulator produces results, agent continues
    5. Record turn metrics (latency, tokens)
    6. Check termination conditions (max turns, goal achieved, error)
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        agent_persona: AgentPersona,
        user_simulator: UserSimulator,
        tool_simulator: ToolSimulator,
        environment: SimulationEnvironment,
        adversarial: AdversarialStrategy | NoOpAdversarial | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.agent = agent_persona
        self.user_sim = user_simulator
        self.tool_sim = tool_simulator
        self.env = environment
        self.adversarial = adversarial or NoOpAdversarial()

    async def run(self) -> ConversationResult:
        """Execute the full multi-turn conversation."""
        turns: list[Turn] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0
        status = "completed"
        error_message: str | None = None

        logger.info(
            "scenario_run_started",
            agent=self.agent.name,
            max_turns=self.env.max_turns,
        )

        try:
            for turn_index in range(self.env.max_turns):
                # === USER TURN ===
                if self.adversarial.should_inject(turn_index):
                    user_message = self.adversarial.generate_adversarial_input(turn_index)
                    logger.debug("adversarial_injected", turn=turn_index)
                else:
                    user_message = await self.user_sim.generate(turns, turn_index)

                user_turn = Turn(role="user", content=user_message)
                turns.append(user_turn)

                # Check for goal/frustration signals
                if "[GOAL_ACHIEVED]" in user_message:
                    status = "goal_achieved"
                    logger.info("goal_achieved", turn=turn_index)
                    break
                if "[FRUSTRATED]" in user_message:
                    status = "frustrated"
                    logger.info("user_frustrated", turn=turn_index)
                    break

                # === AGENT TURN ===
                start_time = time.perf_counter()

                agent_response = await self._agent_turn(turns)

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                # Handle tool calls if present
                if agent_response.tool_calls:
                    tool_results = await self._handle_tool_calls(agent_response.tool_calls)

                    # Record tool call turn
                    tool_turn = Turn(
                        role="assistant",
                        content=agent_response.content,
                        tool_calls=agent_response.tool_calls,
                        tool_results=tool_results,
                        latency_ms=latency_ms,
                        input_tokens=agent_response.input_tokens,
                        output_tokens=agent_response.output_tokens,
                    )
                    turns.append(tool_turn)

                    # Agent follow-up after tool results
                    followup_start = time.perf_counter()
                    followup = await self._agent_turn_with_tool_results(
                        turns, tool_results
                    )
                    followup_latency = int(
                        (time.perf_counter() - followup_start) * 1000
                    )

                    followup_turn = Turn(
                        role="assistant",
                        content=followup.content,
                        latency_ms=followup_latency,
                        input_tokens=followup.input_tokens,
                        output_tokens=followup.output_tokens,
                    )
                    turns.append(followup_turn)

                    total_input_tokens += agent_response.input_tokens + followup.input_tokens
                    total_output_tokens += agent_response.output_tokens + followup.output_tokens
                    total_latency_ms += latency_ms + followup_latency
                else:
                    # No tool calls — straightforward response
                    agent_turn = Turn(
                        role="assistant",
                        content=agent_response.content,
                        latency_ms=latency_ms,
                        input_tokens=agent_response.input_tokens,
                        output_tokens=agent_response.output_tokens,
                    )
                    turns.append(agent_turn)

                    total_input_tokens += agent_response.input_tokens
                    total_output_tokens += agent_response.output_tokens
                    total_latency_ms += latency_ms

                # Check token budget
                total_tokens = total_input_tokens + total_output_tokens
                if total_tokens >= self.env.max_total_tokens:
                    logger.info("token_budget_exceeded", total=total_tokens)
                    break

        except Exception as e:
            status = "failed"
            error_message = str(e)
            logger.error("scenario_run_failed", error=str(e))

        total_tokens = total_input_tokens + total_output_tokens
        turn_count = sum(1 for t in turns if t.role == "user")

        logger.info(
            "scenario_run_completed",
            status=status,
            turn_count=turn_count,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
        )

        return ConversationResult(
            turns=turns,
            turn_count=turn_count,
            total_tokens=total_tokens,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_latency_ms=total_latency_ms,
            status=status,
            error_message=error_message,
        )

    async def _agent_turn(self, turns: list[Turn]) -> Any:
        """Single agent LLM call."""
        messages = self._turns_to_messages(turns)

        # Build tools in OpenAI format if agent has tools
        tools = None
        if self.agent.tools:
            tools = [
                {
                    "type": "function",
                    "function": tool,
                }
                for tool in self.agent.tools
            ]

        return await self.llm_client.chat(
            model=self.agent.model,
            messages=messages,
            system=self.agent.system_prompt,
            tools=tools,
            temperature=self.agent.temperature,
            max_tokens=self.agent.max_tokens,
        )

    async def _agent_turn_with_tool_results(
        self,
        turns: list[Turn],
        tool_results: list[ToolResult],
    ) -> Any:
        """Agent follow-up call after receiving tool results."""
        messages = self._turns_to_messages(turns)

        # Append tool results as tool messages
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            })

        return await self.llm_client.chat(
            model=self.agent.model,
            messages=messages,
            system=self.agent.system_prompt,
            temperature=self.agent.temperature,
            max_tokens=self.agent.max_tokens,
        )

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute all tool calls through the simulator."""
        results = []
        for tc in tool_calls:
            result = await self.tool_sim.execute(tc)
            results.append(result)
        return results

    @staticmethod
    def _turns_to_messages(turns: list[Turn]) -> list[dict[str, Any]]:
        """Convert internal Turn objects to LLM message format."""
        messages: list[dict[str, Any]] = []
        for turn in turns:
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": turn.content}
                if turn.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in turn.tool_calls
                    ]
                messages.append(msg)
        return messages
