from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.models.eval_run import EvalRun
from app.schemas.eval_run import EvalRunCreate, EvalRunListResponse, EvalRunResponse
from app.workers.simulation_tasks import run_simulation

router = APIRouter(prefix="/eval-runs", tags=["eval-runs"])


@router.get("", response_model=EvalRunListResponse)
async def list_eval_runs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    status: str | None = None,
    agent_config_id: str | None = None,
    scenario_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> EvalRunListResponse:
    query = select(EvalRun)
    count_query = select(func.count(EvalRun.id))

    if status:
        query = query.where(EvalRun.status == status)
        count_query = count_query.where(EvalRun.status == status)
    if agent_config_id:
        query = query.where(EvalRun.agent_config_id == agent_config_id)
        count_query = count_query.where(EvalRun.agent_config_id == agent_config_id)
    if scenario_id:
        query = query.where(EvalRun.scenario_id == scenario_id)
        count_query = count_query.where(EvalRun.scenario_id == scenario_id)

    total = (await db.execute(count_query)).scalar_one()
    result = await db.execute(
        query.order_by(EvalRun.created_at.desc()).offset(offset).limit(limit)
    )
    items = [EvalRunResponse.model_validate(r) for r in result.scalars().all()]

    return EvalRunListResponse(total=total, offset=offset, limit=limit, items=items)


@router.post("", response_model=EvalRunResponse, status_code=202)
async def create_eval_run(
    body: EvalRunCreate,
    db: AsyncSession = Depends(get_db),
) -> EvalRunResponse:
    """Create and trigger an evaluation run.

    Returns immediately with status=pending. The simulation runs
    asynchronously via a Celery worker.
    """
    eval_run = EvalRun(
        name=body.name,
        agent_config_id=body.agent_config_id,
        scenario_id=body.scenario_id,
        rubric_id=body.rubric_id,
        num_conversations=body.num_conversations,
        config=body.config,
        status="pending",
    )
    db.add(eval_run)
    await db.flush()
    await db.refresh(eval_run)

    # Trigger async simulation task
    run_simulation.delay(eval_run.id)

    return EvalRunResponse.model_validate(eval_run)


@router.get("/{run_id}", response_model=EvalRunResponse)
async def get_eval_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvalRunResponse:
    result = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
    eval_run = result.scalar_one_or_none()
    if not eval_run:
        raise NotFoundError("EvalRun", run_id)
    return EvalRunResponse.model_validate(eval_run)


@router.post("/{run_id}/cancel", status_code=200)
async def cancel_eval_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvalRunResponse:
    result = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
    eval_run = result.scalar_one_or_none()
    if not eval_run:
        raise NotFoundError("EvalRun", run_id)

    if eval_run.status in ("pending", "running_simulation", "running_evaluation"):
        eval_run.status = "cancelled"
        await db.flush()

    return EvalRunResponse.model_validate(eval_run)
