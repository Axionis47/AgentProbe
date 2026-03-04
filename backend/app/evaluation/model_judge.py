"""LLM-as-judge evaluator.

Uses a model to evaluate conversation quality against rubric dimensions.
Forces structured output via tool_use so scores are machine-parseable.
Falls back to content parsing if the model doesn't return a tool call.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from app.config import settings
from app.engine.types import LLMClientProtocol
from app.evaluation.types import EvaluationResult, RubricDimension

logger = structlog.get_logger()


class ModelJudgeEvaluator:
    """Evaluates conversations using an LLM as a judge."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        model: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model or settings.judge_model

    async def evaluate(
        self,
        conversation_turns: list[dict[str, Any]],
        rubric_dimensions: list[RubricDimension],
    ) -> EvaluationResult:
        """Evaluate a conversation transcript against rubric dimensions."""
        system_prompt = self._build_system_prompt(rubric_dimensions)
        messages = self._build_messages(conversation_turns)
        tools = [self._build_scoring_tool(rubric_dimensions)]

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            system=system_prompt,
            tools=tools,
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=2048,
        )

        return self._parse_response(response, rubric_dimensions)

    def _build_system_prompt(self, dimensions: list[RubricDimension]) -> str:
        """Build the system prompt that instructs the judge."""
        dimension_text = "\n".join(
            f"- **{d.name}** (weight={d.weight}): {d.description}\n"
            f"  Criteria: {', '.join(d.criteria)}"
            for d in dimensions
        )

        return (
            "You are an expert conversation evaluator. Your task is to evaluate "
            "an AI assistant's performance in a multi-turn conversation.\n\n"
            "Score each dimension on a 0-10 scale:\n"
            "  0-2: Very poor\n"
            "  3-4: Below average\n"
            "  5-6: Average\n"
            "  7-8: Good\n"
            "  9-10: Excellent\n\n"
            f"Dimensions to evaluate:\n{dimension_text}\n\n"
            "Use the submit_evaluation tool to report your scores. "
            "Provide a brief reasoning for each score."
        )

    def _build_messages(self, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format conversation turns as a transcript for the judge."""
        lines: list[str] = ["## Conversation Transcript\n"]
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
                    result_content = tr.get("content", "")
                    is_error = tr.get("is_error", False)
                    status = "ERROR" if is_error else "OK"
                    lines.append(f"  ← TOOL_RESULT [{status}]: {result_content[:200]}")

        return [{"role": "user", "content": "\n".join(lines)}]

    def _build_scoring_tool(
        self, dimensions: list[RubricDimension],
    ) -> dict[str, Any]:
        """Build an OpenAI-format tool definition for structured scoring output."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for dim in dimensions:
            properties[f"{dim.name}_score"] = {
                "type": "number",
                "description": f"Score for {dim.name} (0-10): {dim.description}",
                "minimum": 0,
                "maximum": 10,
            }
            properties[f"{dim.name}_reasoning"] = {
                "type": "string",
                "description": f"Brief reasoning for {dim.name} score",
            }
            required.extend([f"{dim.name}_score", f"{dim.name}_reasoning"])

        return {
            "type": "function",
            "function": {
                "name": "submit_evaluation",
                "description": "Submit evaluation scores for all dimensions",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _parse_response(
        self,
        response: Any,
        dimensions: list[RubricDimension],
    ) -> EvaluationResult:
        """Extract scores from tool call or fall back to content parsing."""
        scores: dict[str, float] = {}
        reasoning_parts: list[str] = []

        # Try tool call first (preferred — structured output)
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "submit_evaluation":
                    args = tc.arguments
                    for dim in dimensions:
                        score_key = f"{dim.name}_score"
                        reason_key = f"{dim.name}_reasoning"
                        raw_score = args.get(score_key, 5.0)
                        scores[dim.name] = min(10.0, max(0.0, float(raw_score)))
                        reasoning_parts.append(
                            f"{dim.name}: {scores[dim.name]:.1f}/10 — "
                            f"{args.get(reason_key, 'No reasoning provided')}"
                        )
                    break

        # Fallback: parse scores from content text
        if not scores and response.content:
            scores, reasoning_parts = self._parse_content_fallback(
                response.content, dimensions,
            )

        # If still no scores, use defaults
        if not scores:
            for dim in dimensions:
                scores[dim.name] = 5.0
                reasoning_parts.append(f"{dim.name}: 5.0/10 — Could not parse judge output")

        overall = self._weighted_overall(scores, dimensions)

        return EvaluationResult(
            evaluator_type="model_judge",
            scores=scores,
            overall_score=overall,
            reasoning="\n".join(reasoning_parts),
            metadata={"model": self.model},
        )

    def _parse_content_fallback(
        self,
        content: str,
        dimensions: list[RubricDimension],
    ) -> tuple[dict[str, float], list[str]]:
        """Try to extract scores from free-text content."""
        scores: dict[str, float] = {}
        reasoning: list[str] = []

        for dim in dimensions:
            # Look for patterns like "helpfulness: 7" or "helpfulness: 7/10"
            pattern = rf"(?i){re.escape(dim.name)}\s*[:=]\s*(\d+(?:\.\d+)?)"
            match = re.search(pattern, content)
            if match:
                score = min(10.0, max(0.0, float(match.group(1))))
                scores[dim.name] = score
                reasoning.append(f"{dim.name}: {score:.1f}/10 — parsed from content")

        return scores, reasoning

    @staticmethod
    def _weighted_overall(
        scores: dict[str, float],
        dimensions: list[RubricDimension],
    ) -> float:
        """Compute weighted overall score."""
        total_weight = 0.0
        weighted_sum = 0.0

        for dim in dimensions:
            if dim.name in scores:
                weighted_sum += scores[dim.name] * dim.weight
                total_weight += dim.weight

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 2)
