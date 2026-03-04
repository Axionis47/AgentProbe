from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HumanEvaluationCreate(BaseModel):
    conversation_id: str
    rubric_id: str | None = None
    scores: dict[str, Any]
    overall_score: float = Field(..., ge=0.0, le=10.0)
    reasoning: str | None = None
    per_turn_scores: list[dict[str, Any]] | None = None
    evaluator_id: str | None = None


class EvaluationResponse(BaseModel):
    id: str
    conversation_id: str
    evaluator_type: str
    evaluator_id: str | None
    rubric_id: str | None
    scores: dict[str, Any]
    overall_score: float | None
    reasoning: str | None
    per_turn_scores: list[dict[str, Any]] | None
    metadata: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class EvaluationListResponse(BaseModel):
    total: int
    items: list[EvaluationResponse]


# --- Pairwise Comparison Schemas ---


class PairwiseComparisonRequest(BaseModel):
    conversation_id_a: str
    conversation_id_b: str
    rubric_id: str | None = None


class PairwiseComparisonResponse(BaseModel):
    match_id: str
    winner: str
    conversation_id_a: str
    conversation_id_b: str
    reasoning: str
    dimension_preferences: dict[str, str]
    confidence: float
    evaluations: list[EvaluationResponse]


# --- Rankings Schemas ---


class AgentRanking(BaseModel):
    agent_config_id: str
    agent_name: str | None = None
    elo_rating: float
    matches_played: int
    wins: int
    losses: int
    draws: int


class RankingsResponse(BaseModel):
    scenario_id: str | None
    rankings: list[AgentRanking]
    total_matches: int


# --- Reliability Schema ---


class ReliabilityResponse(BaseModel):
    alpha: float
    num_items: int
    num_raters: int
    per_dimension_alpha: dict[str, float]


# --- Calibration Schema ---


class CalibrationResponse(BaseModel):
    pearson_r: float
    spearman_rho: float
    mae: float
    rmse: float
    bias: float
    n: int
    calibration_curve: list[dict[str, Any]]
    per_dimension: dict[str, dict[str, float]]
