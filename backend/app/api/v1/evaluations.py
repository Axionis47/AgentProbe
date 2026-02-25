"""Evaluation API endpoints."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.agent_config import AgentConfig
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.models.evaluation import Evaluation
from app.schemas.evaluation import (
    AgentRanking,
    CalibrationResponse,
    EvaluationResponse,
    HumanEvaluationCreate,
    PairwiseComparisonRequest,
    PairwiseComparisonResponse,
    RankingsResponse,
    ReliabilityResponse,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


# ---------------------------------------------------------------
# Existing: POST /evaluations/human
# ---------------------------------------------------------------


@router.post(
    "/human",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_human_evaluation(
    payload: HumanEvaluationCreate,
    db: AsyncSession = Depends(get_db),
) -> Evaluation:
    """Submit a human evaluation for a conversation."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == payload.conversation_id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {payload.conversation_id} not found",
        )

    evaluation = Evaluation(
        conversation_id=payload.conversation_id,
        evaluator_type="human",
        evaluator_id=payload.evaluator_id,
        rubric_id=payload.rubric_id,
        scores=payload.scores,
        overall_score=payload.overall_score,
        reasoning=payload.reasoning,
        per_turn_scores=payload.per_turn_scores,
    )
    db.add(evaluation)
    await db.flush()
    await db.refresh(evaluation)
    return evaluation


# ---------------------------------------------------------------
# POST /evaluations/pairwise
# ---------------------------------------------------------------


@router.post(
    "/pairwise",
    response_model=PairwiseComparisonResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_pairwise_comparison(
    payload: PairwiseComparisonRequest,
    db: AsyncSession = Depends(get_db),
) -> PairwiseComparisonResponse:
    """Run a pairwise comparison between two conversations."""
    from app.engine.llm_client import LLMClient
    from app.evaluation.pairwise_judge import PairwiseJudgeEvaluator
    from app.evaluation.types import DEFAULT_DIMENSIONS, RubricDimension
    from app.models.rubric import Rubric

    # Load conversations
    result_a = await db.execute(
        select(Conversation).where(Conversation.id == payload.conversation_id_a)
    )
    conv_a = result_a.scalar_one_or_none()
    if not conv_a:
        raise HTTPException(status_code=404, detail=f"Conversation {payload.conversation_id_a} not found")

    result_b = await db.execute(
        select(Conversation).where(Conversation.id == payload.conversation_id_b)
    )
    conv_b = result_b.scalar_one_or_none()
    if not conv_b:
        raise HTTPException(status_code=404, detail=f"Conversation {payload.conversation_id_b} not found")

    # Load dimensions
    dimensions = DEFAULT_DIMENSIONS
    if payload.rubric_id:
        result = await db.execute(select(Rubric).where(Rubric.id == payload.rubric_id))
        rubric = result.scalar_one_or_none()
        if rubric and rubric.dimensions:
            dimensions = [
                RubricDimension(
                    name=d["name"], description=d.get("description", ""),
                    weight=d.get("weight", 1.0), criteria=d.get("criteria", []),
                )
                for d in rubric.dimensions
            ]

    # Run comparison
    evaluator = PairwiseJudgeEvaluator(llm_client=LLMClient())
    comparison = await evaluator.compare(
        conv_a.turns or [], conv_b.turns or [], dimensions,
    )

    # Map winner to per-conversation result
    result_map = {
        "a": ("win", "loss"),
        "b": ("loss", "win"),
        "draw": ("draw", "draw"),
    }
    result_a_str, result_b_str = result_map.get(comparison.winner, ("draw", "draw"))

    # Store Evaluation on both conversations
    eval_records = []
    for conv, result_str in [(conv_a, result_a_str), (conv_b, result_b_str)]:
        opponent_id = payload.conversation_id_b if conv.id == payload.conversation_id_a else payload.conversation_id_a
        eval_record = Evaluation(
            conversation_id=conv.id,
            evaluator_type="pairwise_judge",
            rubric_id=payload.rubric_id,
            scores=comparison.dimension_preferences,
            overall_score=comparison.confidence * 10.0 if result_str == "win" else (5.0 if result_str == "draw" else (1.0 - comparison.confidence) * 10.0),
            reasoning=comparison.reasoning,
            metadata_={
                "match_id": comparison.match_id,
                "opponent_conversation_id": opponent_id,
                "result": result_str,
                "winner": comparison.winner,
            },
        )
        db.add(eval_record)
        await db.flush()
        await db.refresh(eval_record)
        eval_records.append(eval_record)

    return PairwiseComparisonResponse(
        match_id=comparison.match_id,
        winner=comparison.winner,
        conversation_id_a=payload.conversation_id_a,
        conversation_id_b=payload.conversation_id_b,
        reasoning=comparison.reasoning,
        dimension_preferences=comparison.dimension_preferences,
        confidence=comparison.confidence,
        evaluations=[EvaluationResponse.model_validate(e) for e in eval_records],
    )


# ---------------------------------------------------------------
# GET /evaluations/rankings
# ---------------------------------------------------------------


@router.get("/rankings", response_model=RankingsResponse)
async def get_rankings(
    scenario_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> RankingsResponse:
    """Compute ELO rankings from all pairwise evaluations."""
    from app.evaluation.elo import compute_rankings

    # Load all pairwise evaluations
    query = select(Evaluation).where(Evaluation.evaluator_type == "pairwise_judge")
    result = await db.execute(query)
    all_evals = result.scalars().all()

    # Group by match_id, build match results
    matches_by_id: dict[str, list[Evaluation]] = defaultdict(list)
    for ev in all_evals:
        meta = ev.metadata_ or {}
        match_id = meta.get("match_id")
        if match_id:
            matches_by_id[match_id].append(ev)

    # Resolve agent_config_id for each conversation
    conv_to_agent: dict[str, str] = {}
    match_results: list[dict] = []

    for match_id, evals in matches_by_id.items():
        if len(evals) != 2:
            continue

        for ev in evals:
            cid = ev.conversation_id
            if cid not in conv_to_agent:
                conv_result = await db.execute(
                    select(Conversation.eval_run_id).where(Conversation.id == cid)
                )
                eval_run_id = conv_result.scalar_one_or_none()
                if eval_run_id:
                    run_result = await db.execute(
                        select(EvalRun.agent_config_id).where(EvalRun.id == eval_run_id)
                    )
                    agent_id = run_result.scalar_one_or_none()
                    if agent_id:
                        conv_to_agent[cid] = agent_id

        # Determine agents for this match
        ev_a, ev_b = evals[0], evals[1]
        agent_a = conv_to_agent.get(ev_a.conversation_id)
        agent_b = conv_to_agent.get(ev_b.conversation_id)
        if not agent_a or not agent_b:
            continue

        # Filter by scenario if requested
        if scenario_id:
            run_a = await db.execute(
                select(EvalRun.scenario_id).where(
                    EvalRun.id == (
                        select(Conversation.eval_run_id).where(
                            Conversation.id == ev_a.conversation_id
                        ).scalar_subquery()
                    )
                )
            )
            s_id = run_a.scalar_one_or_none()
            if s_id != scenario_id:
                continue

        meta_a = ev_a.metadata_ or {}
        result_a = meta_a.get("result", "draw")

        if result_a == "win":
            match_results.append({"agent_config_id_a": agent_a, "agent_config_id_b": agent_b, "result": "a_wins"})
        elif result_a == "loss":
            match_results.append({"agent_config_id_a": agent_a, "agent_config_id_b": agent_b, "result": "b_wins"})
        else:
            match_results.append({"agent_config_id_a": agent_a, "agent_config_id_b": agent_b, "result": "draw"})

    # Compute ELO
    ratings = compute_rankings(match_results)

    # Count stats and get names
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "total": 0})
    for mr in match_results:
        a, b, r = mr["agent_config_id_a"], mr["agent_config_id_b"], mr["result"]
        stats[a]["total"] += 1
        stats[b]["total"] += 1
        if r == "a_wins":
            stats[a]["wins"] += 1
            stats[b]["losses"] += 1
        elif r == "b_wins":
            stats[b]["wins"] += 1
            stats[a]["losses"] += 1
        else:
            stats[a]["draws"] += 1
            stats[b]["draws"] += 1

    # Get agent names
    rankings = []
    for agent_id, rating in sorted(ratings.items(), key=lambda x: -x[1]):
        name_result = await db.execute(
            select(AgentConfig.name).where(AgentConfig.id == agent_id)
        )
        name = name_result.scalar_one_or_none()
        s = stats[agent_id]
        rankings.append(AgentRanking(
            agent_config_id=agent_id,
            agent_name=name,
            elo_rating=rating,
            matches_played=s["total"],
            wins=s["wins"],
            losses=s["losses"],
            draws=s["draws"],
        ))

    return RankingsResponse(
        scenario_id=scenario_id,
        rankings=rankings,
        total_matches=len(match_results),
    )


# ---------------------------------------------------------------
# GET /evaluations/reliability
# ---------------------------------------------------------------


@router.get("/reliability", response_model=ReliabilityResponse)
async def get_reliability(
    eval_run_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> ReliabilityResponse:
    """Compute interrater reliability for human evaluations in an eval run."""
    from app.evaluation.reliability import compute_reliability

    # Get all conversation IDs for this run
    conv_result = await db.execute(
        select(Conversation.id).where(Conversation.eval_run_id == eval_run_id)
    )
    conv_ids = [row[0] for row in conv_result.all()]

    if not conv_ids:
        raise HTTPException(status_code=404, detail="No conversations found for this run")

    # Load human evaluations
    eval_result = await db.execute(
        select(Evaluation)
        .where(Evaluation.conversation_id.in_(conv_ids))
        .where(Evaluation.evaluator_type == "human")
        .order_by(Evaluation.created_at)
    )
    evaluations = eval_result.scalars().all()

    # Group by conversation
    by_conv: dict[str, list[dict[str, float]]] = defaultdict(list)
    all_dims: set[str] = set()
    for ev in evaluations:
        scores = ev.scores or {}
        by_conv[ev.conversation_id].append(scores)
        all_dims.update(scores.keys())

    if not by_conv:
        raise HTTPException(status_code=404, detail="No human evaluations found for this run")

    dimensions = sorted(all_dims)
    result = compute_reliability(dict(by_conv), dimensions)

    return ReliabilityResponse(
        alpha=result.alpha,
        num_items=result.num_items,
        num_raters=result.num_raters,
        per_dimension_alpha=result.per_dimension_alpha,
    )


# ---------------------------------------------------------------
# GET /evaluations/calibration
# ---------------------------------------------------------------


@router.get("/calibration", response_model=CalibrationResponse)
async def get_calibration(
    eval_run_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> CalibrationResponse:
    """Compare model judge vs human scores for conversations in an eval run."""
    from app.evaluation.calibration import calibration_curve, calibration_metrics

    # Get all conversation IDs
    conv_result = await db.execute(
        select(Conversation.id).where(Conversation.eval_run_id == eval_run_id)
    )
    conv_ids = [row[0] for row in conv_result.all()]

    if not conv_ids:
        raise HTTPException(status_code=404, detail="No conversations found for this run")

    # Load all evaluations
    eval_result = await db.execute(
        select(Evaluation)
        .where(Evaluation.conversation_id.in_(conv_ids))
        .where(Evaluation.evaluator_type.in_(["human", "model_judge"]))
    )
    evaluations = eval_result.scalars().all()

    # Group by conversation → type
    by_conv: dict[str, dict[str, Evaluation]] = defaultdict(dict)
    for ev in evaluations:
        # If multiple of same type, keep the latest
        by_conv[ev.conversation_id][ev.evaluator_type] = ev

    # Build paired lists
    human_scores: list[float] = []
    model_scores: list[float] = []
    all_dims: set[str] = set()
    dim_human: dict[str, list[float]] = defaultdict(list)
    dim_model: dict[str, list[float]] = defaultdict(list)

    for conv_id, evals_map in by_conv.items():
        if "human" in evals_map and "model_judge" in evals_map:
            h = evals_map["human"]
            m = evals_map["model_judge"]
            if h.overall_score is not None and m.overall_score is not None:
                human_scores.append(h.overall_score)
                model_scores.append(m.overall_score)

            # Per-dimension
            h_scores = h.scores or {}
            m_scores = m.scores or {}
            for dim in set(h_scores.keys()) & set(m_scores.keys()):
                try:
                    dim_human[dim].append(float(h_scores[dim]))
                    dim_model[dim].append(float(m_scores[dim]))
                    all_dims.add(dim)
                except (ValueError, TypeError):
                    pass

    if len(human_scores) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 paired human+model evaluations, found {len(human_scores)}",
        )

    # Overall metrics
    metrics = calibration_metrics(human_scores, model_scores)
    curve = calibration_curve(human_scores, model_scores, num_bins=10)

    # Per-dimension metrics
    per_dim: dict[str, dict[str, float]] = {}
    for dim in sorted(all_dims):
        if len(dim_human[dim]) >= 2:
            dm = calibration_metrics(dim_human[dim], dim_model[dim])
            per_dim[dim] = {
                "pearson_r": dm.pearson_r,
                "spearman_rho": dm.spearman_rho,
                "mae": dm.mae,
                "bias": dm.bias,
            }

    return CalibrationResponse(
        pearson_r=metrics.pearson_r,
        spearman_rho=metrics.spearman_rho,
        mae=metrics.mae,
        rmse=metrics.rmse,
        bias=metrics.bias,
        n=metrics.n,
        calibration_curve=[
            {"bin_center": b.bin_center, "avg_human": b.avg_human, "avg_model": b.avg_model, "count": b.count}
            for b in curve
        ],
        per_dimension=per_dim,
    )
