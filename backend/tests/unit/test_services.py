"""Unit tests for the service layer (AgentSimulationService, EvaluationService).

All LLM calls, DB operations, and Kafka events are mocked.
Tests verify orchestration logic, not LLM quality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.engine.types import ConversationResult, LLMResponse, ToolCall, Turn
from app.evaluation.types import DEFAULT_DIMENSIONS, EvaluationResult, MetricValue
from app.services.agent_simulation import AgentSimulationService
from app.services.evaluation_service import EvaluationService


# ============================================================
# Helpers
# ============================================================


def _make_mock_eval_run(**overrides):
    run = MagicMock()
    run.id = overrides.get("id", "run-001")
    run.agent_config_id = overrides.get("agent_config_id", "cfg-001")
    run.scenario_id = overrides.get("scenario_id", "scn-001")
    run.rubric_id = overrides.get("rubric_id", None)
    run.num_conversations = overrides.get("num_conversations", 2)
    run.status = overrides.get("status", "pending")
    run.config = overrides.get("config", {})
    run.started_at = None
    run.completed_at = None
    run.error_message = None
    return run


def _make_mock_agent_config(**overrides):
    cfg = MagicMock()
    cfg.name = overrides.get("name", "Test Agent")
    cfg.system_prompt = overrides.get("system_prompt", "Be helpful.")
    cfg.model = overrides.get("model", "test-model")
    cfg.temperature = overrides.get("temperature", 0.7)
    cfg.max_tokens = overrides.get("max_tokens", 4096)
    cfg.tools = overrides.get("tools", [])
    return cfg


def _make_mock_scenario(**overrides):
    scn = MagicMock()
    scn.user_persona = overrides.get("user_persona", {"personality": "neutral", "goal": "Get help"})
    scn.constraints = overrides.get("constraints", {"max_turns": 3})
    scn.turns_template = overrides.get("turns_template", [{"role": "user", "content": "Hello"}])
    return scn


def _make_mock_conversation(**overrides):
    conv = MagicMock()
    conv.id = overrides.get("id", "conv-001")
    conv.eval_run_id = overrides.get("eval_run_id", "run-001")
    conv.turns = overrides.get("turns", [
        {"role": "user", "content": "Help me"},
        {"role": "assistant", "content": "Sure!", "latency_ms": 100, "input_tokens": 10, "output_tokens": 20},
    ])
    conv.turn_count = overrides.get("turn_count", 1)
    conv.total_tokens = overrides.get("total_tokens", 30)
    conv.total_input_tokens = overrides.get("total_input_tokens", 10)
    conv.total_output_tokens = overrides.get("total_output_tokens", 20)
    conv.total_latency_ms = overrides.get("total_latency_ms", 100)
    conv.status = overrides.get("status", "completed")
    return conv


def _make_judge_response():
    """Mock LLM response for model judge."""
    args = {}
    for dim in DEFAULT_DIMENSIONS:
        args[f"{dim.name}_score"] = 7.0
        args[f"{dim.name}_reasoning"] = f"Good {dim.name}"
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="call_1", name="submit_evaluation", arguments=args)],
        input_tokens=50,
        output_tokens=100,
        model="test-model",
        stop_reason="tool_calls",
    )


def _setup_db_for_simulation(mock_db, eval_run, agent_config, scenario):
    """Configure mock_db.execute to return run, config, scenario in sequence."""
    run_result = MagicMock()
    run_result.scalar_one_or_none.return_value = eval_run
    run_result.scalar_one.return_value = eval_run

    cfg_result = MagicMock()
    cfg_result.scalar_one_or_none.return_value = agent_config
    cfg_result.scalar_one.return_value = agent_config

    scn_result = MagicMock()
    scn_result.scalar_one_or_none.return_value = scenario
    scn_result.scalar_one.return_value = scenario

    mock_db.execute = AsyncMock(side_effect=[run_result, cfg_result, scn_result])


# ============================================================
# AgentSimulationService
# ============================================================


class TestAgentSimulationService:
    """Tests for AgentSimulationService.run_eval orchestration."""

    @pytest.mark.asyncio
    async def test_run_eval_sets_status_to_running(self):
        """run_eval transitions status from pending to running_simulation."""
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()

        eval_run = _make_mock_eval_run(num_conversations=0)
        agent_config = _make_mock_agent_config()
        scenario = _make_mock_scenario()

        _setup_db_for_simulation(mock_db, eval_run, agent_config, scenario)

        service = AgentSimulationService(db=mock_db)
        with patch.object(service, "llm_client"):
            await service.run_eval("run-001")

        # Status should have been set to running_simulation at some point
        assert eval_run.started_at is not None

    @pytest.mark.asyncio
    async def test_run_eval_not_found_raises(self):
        """run_eval raises ValueError if eval run not found."""
        mock_db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result)

        service = AgentSimulationService(db=mock_db)

        with pytest.raises(ValueError, match="not found"):
            await service.run_eval("nonexistent-run")

    @pytest.mark.asyncio
    async def test_run_eval_failure_sets_failed_status(self):
        """If simulation fails, status is set to failed with error message."""
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()

        eval_run = _make_mock_eval_run(num_conversations=1)
        agent_config = _make_mock_agent_config()
        scenario = _make_mock_scenario()

        _setup_db_for_simulation(mock_db, eval_run, agent_config, scenario)

        service = AgentSimulationService(db=mock_db)

        # Make the LLM client raise an exception during conversation
        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(side_effect=Exception("LLM API down"))
            await service.run_eval("run-001")

        # The service catches conversation failures and marks the run as failed
        assert eval_run.status == "failed"

    @pytest.mark.asyncio
    async def test_run_eval_with_zero_conversations(self):
        """run_eval with num_conversations=0 completes immediately."""
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()

        eval_run = _make_mock_eval_run(num_conversations=0)
        agent_config = _make_mock_agent_config()
        scenario = _make_mock_scenario()

        _setup_db_for_simulation(mock_db, eval_run, agent_config, scenario)

        service = AgentSimulationService(db=mock_db)
        with patch.object(service, "llm_client"):
            await service.run_eval("run-001")

        # With zero conversations and no successes, pipeline treats this as failed
        assert eval_run.status == "failed"


# ============================================================
# EvaluationService
# ============================================================


class TestEvaluationServiceOrchestration:
    """Tests for EvaluationService.evaluate_conversation orchestration."""

    @pytest.mark.asyncio
    async def test_evaluate_creates_judge_and_grader_evaluations(self):
        """evaluate_conversation creates both model_judge and rubric_grader evaluations."""
        mock_db = AsyncMock()
        mock_conv = _make_mock_conversation()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        service = EvaluationService(db=mock_db)

        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_judge_response())
            evaluations = await service.evaluate_conversation("conv-001")

        # Should have at least 2 evaluations (judge + grader)
        assert len(evaluations) >= 2

    @pytest.mark.asyncio
    async def test_evaluate_conversation_not_found_raises(self):
        """evaluate_conversation raises ValueError if conversation not found."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        service = EvaluationService(db=mock_db)

        with pytest.raises(ValueError, match="not found"):
            await service.evaluate_conversation("nonexistent-conv")

    @pytest.mark.asyncio
    async def test_default_dimensions_used_when_no_rubric(self):
        """When rubric_id is None, DEFAULT_DIMENSIONS are used."""
        service = EvaluationService(db=AsyncMock())
        dimensions = await service._load_dimensions(None)
        assert dimensions == DEFAULT_DIMENSIONS
        assert len(dimensions) == 5

    @pytest.mark.asyncio
    async def test_load_dimensions_from_rubric(self):
        """When rubric_id is provided, dimensions are loaded from DB."""
        mock_db = AsyncMock()
        mock_rubric = MagicMock()
        mock_rubric.dimensions = [
            {"name": "custom_dim", "description": "Custom", "weight": 1.0, "criteria": ["c1"]},
        ]
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_rubric
        mock_db.execute = AsyncMock(return_value=mock_result)

        service = EvaluationService(db=mock_db)
        dimensions = await service._load_dimensions("rubric-001")

        assert len(dimensions) == 1
        assert dimensions[0].name == "custom_dim"

    @pytest.mark.asyncio
    async def test_load_dimensions_falls_back_for_empty_rubric(self):
        """When rubric has no dimensions, falls back to defaults."""
        mock_db = AsyncMock()
        mock_rubric = MagicMock()
        mock_rubric.dimensions = []
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_rubric
        mock_db.execute = AsyncMock(return_value=mock_result)

        service = EvaluationService(db=mock_db)
        dimensions = await service._load_dimensions("rubric-empty")

        assert dimensions == DEFAULT_DIMENSIONS

    @pytest.mark.asyncio
    async def test_load_dimensions_falls_back_for_missing_rubric(self):
        """When rubric_id points to nonexistent rubric, falls back to defaults."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        service = EvaluationService(db=mock_db)
        dimensions = await service._load_dimensions("rubric-missing")

        assert dimensions == DEFAULT_DIMENSIONS

    @pytest.mark.asyncio
    async def test_metrics_stored_via_db_add(self):
        """Automated metrics are stored by calling db.add() for each metric."""
        mock_db = AsyncMock()
        mock_conv = _make_mock_conversation()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        service = EvaluationService(db=mock_db)

        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_judge_response())
            await service.evaluate_conversation("conv-001")

        # db.add should be called for evaluations + metrics
        assert mock_db.add.call_count > 0

    @pytest.mark.asyncio
    async def test_judge_failure_does_not_block_grader(self):
        """If model judge fails, rubric grader still runs."""
        mock_db = AsyncMock()
        mock_conv = _make_mock_conversation()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        service = EvaluationService(db=mock_db)

        # Make the LLM client fail (model judge uses LLM)
        with patch.object(service, "llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(side_effect=Exception("LLM timeout"))
            evaluations = await service.evaluate_conversation("conv-001")

        # Grader should still produce an evaluation even if judge fails
        # (grader doesn't use LLM client)
        assert len(evaluations) >= 1


class TestEvaluationServiceHelpers:
    """Tests for static helper methods on EvaluationService."""

    def test_has_reference_answers_true(self):
        """Returns True when turns have expected_response."""
        scenario = MagicMock()
        scenario.turns_template = [
            {"role": "user", "content": "Q1", "expected_response": "A1"},
        ]
        assert EvaluationService._has_reference_answers(scenario) is True

    def test_has_reference_answers_false(self):
        """Returns False when no turns have expected_response."""
        scenario = MagicMock()
        scenario.turns_template = [
            {"role": "user", "content": "Q1"},
        ]
        assert EvaluationService._has_reference_answers(scenario) is False

    def test_has_reference_answers_empty_template(self):
        """Returns False when turns_template is empty."""
        scenario = MagicMock()
        scenario.turns_template = []
        assert EvaluationService._has_reference_answers(scenario) is False

    def test_has_reference_answers_none_template(self):
        """Returns False when turns_template is None."""
        scenario = MagicMock()
        scenario.turns_template = None
        assert EvaluationService._has_reference_answers(scenario) is False

    def test_has_expected_trajectory_true(self):
        """Returns True when constraints contain expected_tool_sequence."""
        scenario = MagicMock()
        scenario.constraints = {"expected_tool_sequence": ["search", "summarize"]}
        assert EvaluationService._has_expected_trajectory(scenario) is True

    def test_has_expected_trajectory_false_empty_seq(self):
        """Returns False when expected_tool_sequence is empty."""
        scenario = MagicMock()
        scenario.constraints = {"expected_tool_sequence": []}
        assert EvaluationService._has_expected_trajectory(scenario) is False

    def test_has_expected_trajectory_false_no_key(self):
        """Returns False when constraints lack expected_tool_sequence."""
        scenario = MagicMock()
        scenario.constraints = {}
        assert EvaluationService._has_expected_trajectory(scenario) is False

    def test_has_expected_trajectory_none_constraints(self):
        """Returns False when constraints is None."""
        scenario = MagicMock()
        scenario.constraints = None
        assert EvaluationService._has_expected_trajectory(scenario) is False

    def test_enrich_turns_with_references(self):
        """Enriches actual turns with expected_response from template."""
        scenario = MagicMock()
        scenario.turns_template = [
            {"role": "user", "content": "Q1", "expected_response": "A1"},
            {"role": "assistant", "content": ""},
        ]
        actual = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "Generated A1"},
        ]

        result = EvaluationService._enrich_turns_with_references(actual, scenario)

        assert result[0]["expected_response"] == "A1"
        assert "expected_response" not in result[1]

    def test_enrich_turns_shorter_template(self):
        """When template is shorter than actual turns, extra turns are unchanged."""
        scenario = MagicMock()
        scenario.turns_template = [{"role": "user", "content": "Q1", "expected_response": "A1"}]
        actual = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Followup"},
        ]

        result = EvaluationService._enrich_turns_with_references(actual, scenario)

        assert len(result) == 3
        assert result[0]["expected_response"] == "A1"
        assert "expected_response" not in result[2]
