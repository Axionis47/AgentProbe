"""Unit tests for Pydantic schema validation.

Tests valid/invalid inputs, enum validation, default values,
and JSONB field validation for all Create/Update schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.agent_config import AgentConfigCreate, AgentConfigUpdate
from app.schemas.common import (
    ConversationStatus,
    Difficulty,
    EvalRunStatus,
    EvaluatorType,
    PaginationParams,
)
from app.schemas.eval_run import EvalRunCreate
from app.schemas.evaluation import HumanEvaluationCreate, PairwiseComparisonRequest
from app.schemas.rubric import RubricCreate, RubricUpdate
from app.schemas.scenario import ScenarioCreate, ScenarioUpdate


# ============================================================
# AgentConfigCreate
# ============================================================


class TestAgentConfigCreateSchema:
    """Validate AgentConfigCreate input constraints."""

    def test_valid_minimal_input(self):
        """Minimal valid input requires name and system_prompt."""
        cfg = AgentConfigCreate(name="Agent", system_prompt="Be helpful.")
        assert cfg.name == "Agent"
        assert cfg.model == "claude-sonnet-4-20250514"  # default
        assert cfg.temperature == 0.7  # default
        assert cfg.max_tokens == 4096  # default
        assert cfg.tools == []
        assert cfg.metadata == {}

    def test_valid_full_input(self):
        """All fields provided."""
        cfg = AgentConfigCreate(
            name="Full Agent",
            description="Description",
            system_prompt="Prompt",
            model="gpt-4",
            temperature=1.5,
            max_tokens=8000,
            tools=[{"name": "search", "description": "Search"}],
            metadata={"team": "ml"},
        )
        assert cfg.temperature == 1.5
        assert len(cfg.tools) == 1

    def test_empty_name_rejected(self):
        """Empty name string is rejected (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfigCreate(name="", system_prompt="Prompt")
        assert "name" in str(exc_info.value)

    def test_name_too_long_rejected(self):
        """Name exceeding 255 characters is rejected."""
        with pytest.raises(ValidationError):
            AgentConfigCreate(name="x" * 256, system_prompt="Prompt")

    def test_empty_system_prompt_rejected(self):
        """Empty system_prompt is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            AgentConfigCreate(name="Agent", system_prompt="")

    def test_temperature_below_zero_rejected(self):
        """Temperature below 0.0 is rejected."""
        with pytest.raises(ValidationError):
            AgentConfigCreate(name="Agent", system_prompt="P", temperature=-0.1)

    def test_temperature_above_two_rejected(self):
        """Temperature above 2.0 is rejected."""
        with pytest.raises(ValidationError):
            AgentConfigCreate(name="Agent", system_prompt="P", temperature=2.1)

    def test_max_tokens_below_one_rejected(self):
        """max_tokens below 1 is rejected."""
        with pytest.raises(ValidationError):
            AgentConfigCreate(name="Agent", system_prompt="P", max_tokens=0)

    def test_max_tokens_above_limit_rejected(self):
        """max_tokens above 200000 is rejected."""
        with pytest.raises(ValidationError):
            AgentConfigCreate(name="Agent", system_prompt="P", max_tokens=200001)

    def test_temperature_boundary_zero(self):
        """Temperature at 0.0 is accepted."""
        cfg = AgentConfigCreate(name="Agent", system_prompt="P", temperature=0.0)
        assert cfg.temperature == 0.0

    def test_temperature_boundary_two(self):
        """Temperature at 2.0 is accepted."""
        cfg = AgentConfigCreate(name="Agent", system_prompt="P", temperature=2.0)
        assert cfg.temperature == 2.0


# ============================================================
# AgentConfigUpdate
# ============================================================


class TestAgentConfigUpdateSchema:
    """Validate AgentConfigUpdate allows partial updates."""

    def test_all_none_is_valid(self):
        """An update with no fields set is valid (no-op)."""
        upd = AgentConfigUpdate()
        assert upd.name is None
        assert upd.model is None

    def test_partial_update_name_only(self):
        """Updating just the name is valid."""
        upd = AgentConfigUpdate(name="New Name")
        assert upd.name == "New Name"
        assert upd.temperature is None

    def test_update_temperature_validation(self):
        """Update temperature is also validated against bounds."""
        with pytest.raises(ValidationError):
            AgentConfigUpdate(temperature=3.0)


# ============================================================
# ScenarioCreate
# ============================================================


class TestScenarioCreateSchema:
    """Validate ScenarioCreate input constraints."""

    def test_valid_minimal_input(self):
        """Minimal valid input requires name and turns_template."""
        s = ScenarioCreate(
            name="Scenario",
            turns_template=[{"role": "user", "content": "Hi"}],
        )
        assert s.difficulty == "medium"  # default
        assert s.tags == []
        assert s.user_persona == {}
        assert s.constraints == {}

    def test_empty_name_rejected(self):
        """Empty name is rejected."""
        with pytest.raises(ValidationError):
            ScenarioCreate(name="", turns_template=[{"role": "user", "content": "Hi"}])

    def test_empty_turns_template_rejected(self):
        """Empty turns_template list is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            ScenarioCreate(name="S", turns_template=[])

    def test_turns_template_accepts_any_dict(self):
        """turns_template accepts arbitrary dict structures."""
        s = ScenarioCreate(
            name="S",
            turns_template=[{"custom_key": "value", "nested": {"a": 1}}],
        )
        assert len(s.turns_template) == 1

    def test_user_persona_accepts_arbitrary_dict(self):
        """user_persona JSONB field accepts any dict."""
        s = ScenarioCreate(
            name="S",
            turns_template=[{"role": "user", "content": "Hi"}],
            user_persona={"custom_field": True, "nested": [1, 2, 3]},
        )
        assert s.user_persona["custom_field"] is True


# ============================================================
# ScenarioUpdate
# ============================================================


class TestScenarioUpdateSchema:
    """Validate ScenarioUpdate allows partial updates."""

    def test_all_none_is_valid(self):
        """An update with no fields set is valid."""
        upd = ScenarioUpdate()
        assert upd.name is None

    def test_update_difficulty_only(self):
        """Updating just difficulty is valid."""
        upd = ScenarioUpdate(difficulty="hard")
        assert upd.difficulty == "hard"

    def test_update_tags(self):
        """Updating tags replaces them."""
        upd = ScenarioUpdate(tags=["new_tag", "another"])
        assert upd.tags == ["new_tag", "another"]


# ============================================================
# EvalRunCreate
# ============================================================


class TestEvalRunCreateSchema:
    """Validate EvalRunCreate input constraints."""

    def test_valid_minimal_input(self):
        """Minimal valid input requires agent_config_id and scenario_id."""
        run = EvalRunCreate(
            agent_config_id="cfg-001",
            scenario_id="scn-001",
        )
        assert run.num_conversations == 5  # default
        assert run.config == {}  # default
        assert run.rubric_id is None

    def test_num_conversations_default(self):
        """Default num_conversations is 5."""
        run = EvalRunCreate(agent_config_id="a", scenario_id="b")
        assert run.num_conversations == 5

    def test_num_conversations_below_one_rejected(self):
        """num_conversations below 1 is rejected."""
        with pytest.raises(ValidationError):
            EvalRunCreate(agent_config_id="a", scenario_id="b", num_conversations=0)

    def test_num_conversations_above_100_rejected(self):
        """num_conversations above 100 is rejected."""
        with pytest.raises(ValidationError):
            EvalRunCreate(agent_config_id="a", scenario_id="b", num_conversations=101)

    def test_num_conversations_boundary_1(self):
        """num_conversations=1 is accepted."""
        run = EvalRunCreate(agent_config_id="a", scenario_id="b", num_conversations=1)
        assert run.num_conversations == 1

    def test_num_conversations_boundary_100(self):
        """num_conversations=100 is accepted."""
        run = EvalRunCreate(agent_config_id="a", scenario_id="b", num_conversations=100)
        assert run.num_conversations == 100

    def test_config_accepts_arbitrary_dict(self):
        """config JSONB field accepts any dict."""
        run = EvalRunCreate(
            agent_config_id="a",
            scenario_id="b",
            config={"custom_key": [1, 2, 3]},
        )
        assert run.config["custom_key"] == [1, 2, 3]

    def test_missing_agent_config_id_rejected(self):
        """Missing agent_config_id is rejected."""
        with pytest.raises(ValidationError):
            EvalRunCreate(scenario_id="b")

    def test_missing_scenario_id_rejected(self):
        """Missing scenario_id is rejected."""
        with pytest.raises(ValidationError):
            EvalRunCreate(agent_config_id="a")


# ============================================================
# RubricCreate
# ============================================================


class TestRubricCreateSchema:
    """Validate RubricCreate input constraints."""

    def test_valid_minimal_input(self):
        """Minimal valid rubric with one dimension."""
        r = RubricCreate(
            name="Rubric",
            dimensions=[{"name": "quality", "description": "Quality check"}],
        )
        assert r.name == "Rubric"
        assert len(r.dimensions) == 1

    def test_empty_dimensions_rejected(self):
        """Empty dimensions list is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            RubricCreate(name="Rubric", dimensions=[])

    def test_empty_name_rejected(self):
        """Empty name is rejected."""
        with pytest.raises(ValidationError):
            RubricCreate(name="", dimensions=[{"name": "d"}])

    def test_dimensions_accept_any_dict(self):
        """Dimensions accept arbitrary dict structures."""
        r = RubricCreate(
            name="R",
            dimensions=[
                {"name": "d1", "weight": 0.5, "criteria": ["a", "b"]},
                {"name": "d2", "custom_field": True},
            ],
        )
        assert len(r.dimensions) == 2


# ============================================================
# RubricUpdate
# ============================================================


class TestRubricUpdateSchema:
    """Validate RubricUpdate allows partial updates."""

    def test_all_none_is_valid(self):
        """An update with no fields is valid."""
        upd = RubricUpdate()
        assert upd.name is None
        assert upd.dimensions is None

    def test_update_name_only(self):
        """Updating just the name is valid."""
        upd = RubricUpdate(name="Updated Rubric")
        assert upd.name == "Updated Rubric"


# ============================================================
# HumanEvaluationCreate
# ============================================================


class TestHumanEvaluationCreateSchema:
    """Validate HumanEvaluationCreate constraints."""

    def test_valid_input(self):
        """Valid human evaluation input."""
        he = HumanEvaluationCreate(
            conversation_id="conv-001",
            scores={"helpfulness": 8.0},
            overall_score=8.0,
        )
        assert he.overall_score == 8.0

    def test_overall_score_below_zero_rejected(self):
        """overall_score below 0.0 is rejected."""
        with pytest.raises(ValidationError):
            HumanEvaluationCreate(
                conversation_id="c",
                scores={},
                overall_score=-0.1,
            )

    def test_overall_score_above_ten_rejected(self):
        """overall_score above 10.0 is rejected."""
        with pytest.raises(ValidationError):
            HumanEvaluationCreate(
                conversation_id="c",
                scores={},
                overall_score=10.1,
            )


# ============================================================
# Enum validation
# ============================================================


class TestEnumValidation:
    """Validate enum classes have expected members."""

    def test_eval_run_status_members(self):
        """EvalRunStatus has all expected members."""
        assert EvalRunStatus.PENDING == "pending"
        assert EvalRunStatus.RUNNING_SIMULATION == "running_simulation"
        assert EvalRunStatus.RUNNING_EVALUATION == "running_evaluation"
        assert EvalRunStatus.COMPLETED == "completed"
        assert EvalRunStatus.FAILED == "failed"
        assert EvalRunStatus.CANCELLED == "cancelled"

    def test_conversation_status_members(self):
        """ConversationStatus has all expected members."""
        assert ConversationStatus.PENDING == "pending"
        assert ConversationStatus.RUNNING == "running"
        assert ConversationStatus.COMPLETED == "completed"
        assert ConversationStatus.FAILED == "failed"

    def test_evaluator_type_members(self):
        """EvaluatorType has all expected members."""
        assert EvaluatorType.MODEL_JUDGE == "model_judge"
        assert EvaluatorType.RUBRIC_GRADER == "rubric_grader"
        assert EvaluatorType.HUMAN == "human"
        assert EvaluatorType.REFERENCE_BASED == "reference_based"
        assert EvaluatorType.TRAJECTORY == "trajectory"
        assert EvaluatorType.PAIRWISE_JUDGE == "pairwise_judge"

    def test_difficulty_members(self):
        """Difficulty has all expected members."""
        assert Difficulty.EASY == "easy"
        assert Difficulty.MEDIUM == "medium"
        assert Difficulty.HARD == "hard"
        assert Difficulty.ADVERSARIAL == "adversarial"

    def test_eval_run_status_is_str_enum(self):
        """EvalRunStatus values can be used as strings."""
        assert EvalRunStatus.PENDING.value == "pending"
        assert str(EvalRunStatus.PENDING) == "EvalRunStatus.PENDING" or EvalRunStatus.PENDING == "pending"

    def test_invalid_enum_value_raises(self):
        """Creating an enum with invalid value raises ValueError."""
        with pytest.raises(ValueError):
            EvalRunStatus("nonexistent_status")


# ============================================================
# PaginationParams
# ============================================================


class TestPaginationParams:
    """Validate pagination parameter constraints."""

    def test_default_values(self):
        """Default offset=0, limit=20."""
        p = PaginationParams()
        assert p.offset == 0
        assert p.limit == 20

    def test_negative_offset_rejected(self):
        """Negative offset is rejected."""
        with pytest.raises(ValidationError):
            PaginationParams(offset=-1)

    def test_limit_below_one_rejected(self):
        """Limit below 1 is rejected."""
        with pytest.raises(ValidationError):
            PaginationParams(limit=0)

    def test_limit_above_100_rejected(self):
        """Limit above 100 is rejected."""
        with pytest.raises(ValidationError):
            PaginationParams(limit=101)

    def test_boundary_values_accepted(self):
        """Boundary values (offset=0, limit=1, limit=100) are accepted."""
        p1 = PaginationParams(offset=0, limit=1)
        assert p1.limit == 1
        p2 = PaginationParams(offset=0, limit=100)
        assert p2.limit == 100
