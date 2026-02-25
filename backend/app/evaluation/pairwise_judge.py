"""Pairwise comparison evaluator — LLM judges which agent performed better.

Does NOT implement EvaluatorProtocol since it takes TWO conversations.
Uses the same LLM calling + tool_use pattern as ModelJudgeEvaluator.
Randomly swaps A/B position to mitigate position bias.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

import structlog
from uuid_extensions import uuid7

from app.config import settings
from app.engine.types import LLMClientProtocol
from app.evaluation.types import RubricDimension

logger = structlog.get_logger()


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison between two conversations."""

    match_id: str
    winner: str  # "a" | "b" | "draw"
    reasoning: str
    dimension_preferences: dict[str, str]  # dimension_name -> "a"|"b"|"draw"
    confidence: float  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PairwiseJudgeEvaluator:
    """Compares two conversations and determines which agent performed better."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        model: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model or settings.judge_model

    async def compare(
        self,
        turns_a: list[dict[str, Any]],
        turns_b: list[dict[str, Any]],
        rubric_dimensions: list[RubricDimension],
    ) -> PairwiseResult:
        """Compare two conversations and return which is better.

        Randomly assigns labels to avoid position bias, then de-shuffles.
        """
        match_id = str(uuid7())
        swapped = random.choice([True, False])

        if swapped:
            presented_a, presented_b = turns_b, turns_a
        else:
            presented_a, presented_b = turns_a, turns_b

        system_prompt, messages = self._build_comparison_prompt(
            presented_a, presented_b, rubric_dimensions,
        )
        tools = [self._build_comparison_tool(rubric_dimensions)]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            system=system_prompt,
            tools=tools,
            temperature=0.1,
            max_tokens=2048,
        )

        result = self._parse_comparison_response(response, rubric_dimensions)

        # De-shuffle: if we swapped, flip the winner label
        if swapped:
            result = self._unswap(result)

        result.match_id = match_id
        result.metadata["model"] = self.model
        result.metadata["swapped"] = swapped

        return result

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_comparison_prompt(
        self,
        turns_a: list[dict[str, Any]],
        turns_b: list[dict[str, Any]],
        dimensions: list[RubricDimension],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build system prompt and messages for comparison."""
        dim_text = "\n".join(
            f"- **{d.name}** (weight={d.weight}): {d.description}"
            for d in dimensions
        )

        system = (
            "You are an expert evaluator comparing two AI assistants. "
            "You will see two conversations (Agent A and Agent B) responding "
            "to the same scenario.\n\n"
            "For each dimension, state your preference (a, b, or draw). "
            "Then give an overall winner.\n\n"
            f"Dimensions:\n{dim_text}\n\n"
            "Use the submit_comparison tool to report your judgment."
        )

        transcript_a = self._format_transcript(turns_a, "Agent A")
        transcript_b = self._format_transcript(turns_b, "Agent B")

        messages = [{
            "role": "user",
            "content": f"{transcript_a}\n\n---\n\n{transcript_b}",
        }]

        return system, messages

    def _build_comparison_tool(
        self,
        dimensions: list[RubricDimension],
    ) -> dict[str, Any]:
        """Build tool schema for structured comparison output."""
        properties: dict[str, Any] = {
            "winner": {
                "type": "string",
                "enum": ["a", "b", "draw"],
                "description": "Overall winner: 'a', 'b', or 'draw'",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence in the judgment (0-1)",
            },
            "reasoning": {
                "type": "string",
                "description": "Overall reasoning for the comparison",
            },
        }
        required = ["winner", "confidence", "reasoning"]

        for dim in dimensions:
            key = f"{dim.name}_preference"
            properties[key] = {
                "type": "string",
                "enum": ["a", "b", "draw"],
                "description": f"Preference for {dim.name}: 'a', 'b', or 'draw'",
            }
            required.append(key)

        return {
            "type": "function",
            "function": {
                "name": "submit_comparison",
                "description": "Submit pairwise comparison judgment",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_comparison_response(
        self,
        response: Any,
        dimensions: list[RubricDimension],
    ) -> PairwiseResult:
        """Parse LLM response into PairwiseResult."""
        winner = "draw"
        reasoning = ""
        confidence = 0.5
        dim_prefs: dict[str, str] = {}

        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "submit_comparison":
                    args = tc.arguments
                    winner = args.get("winner", "draw")
                    confidence = min(1.0, max(0.0, float(args.get("confidence", 0.5))))
                    reasoning = args.get("reasoning", "")

                    for dim in dimensions:
                        key = f"{dim.name}_preference"
                        dim_prefs[dim.name] = args.get(key, "draw")
                    break

        # Fallback: check content
        if not reasoning and response.content:
            reasoning = response.content[:500]

        return PairwiseResult(
            match_id="",  # Will be set by caller
            winner=winner,
            reasoning=reasoning,
            dimension_preferences=dim_prefs,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_transcript(turns: list[dict[str, Any]], label: str) -> str:
        """Format conversation turns as a labeled transcript."""
        lines = [f"## {label}\n"]
        for i, turn in enumerate(turns):
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "")
            lines.append(f"[Turn {i}] {role}: {content}")

            if turn.get("tool_calls"):
                for tc in turn["tool_calls"]:
                    name = tc.get("name", "unknown")
                    args = tc.get("arguments", {})
                    lines.append(f"  → TOOL_CALL: {name}({json.dumps(args)})")

            if turn.get("tool_results"):
                for tr in turn["tool_results"]:
                    status = "ERROR" if tr.get("is_error") else "OK"
                    lines.append(f"  ← TOOL_RESULT [{status}]: {tr.get('content', '')[:200]}")

        return "\n".join(lines)

    @staticmethod
    def _unswap(result: PairwiseResult) -> PairwiseResult:
        """Flip winner labels back when A/B were swapped for position bias."""
        flip = {"a": "b", "b": "a", "draw": "draw"}
        result.winner = flip.get(result.winner, result.winner)
        result.dimension_preferences = {
            k: flip.get(v, v) for k, v in result.dimension_preferences.items()
        }
        return result
