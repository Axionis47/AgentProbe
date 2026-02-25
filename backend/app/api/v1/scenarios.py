from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.models.scenario import Scenario
from app.schemas.scenario import (
    ScenarioCreate,
    ScenarioListResponse,
    ScenarioResponse,
    ScenarioUpdate,
)

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.get("", response_model=ScenarioListResponse)
async def list_scenarios(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    is_active: bool | None = None,
    category: str | None = None,
    difficulty: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> ScenarioListResponse:
    query = select(Scenario)
    count_query = select(func.count(Scenario.id))

    if is_active is not None:
        query = query.where(Scenario.is_active == is_active)
        count_query = count_query.where(Scenario.is_active == is_active)
    if category:
        query = query.where(Scenario.category == category)
        count_query = count_query.where(Scenario.category == category)
    if difficulty:
        query = query.where(Scenario.difficulty == difficulty)
        count_query = count_query.where(Scenario.difficulty == difficulty)

    total = (await db.execute(count_query)).scalar_one()
    result = await db.execute(
        query.order_by(Scenario.created_at.desc()).offset(offset).limit(limit)
    )
    items = [ScenarioResponse.model_validate(r) for r in result.scalars().all()]

    return ScenarioListResponse(total=total, offset=offset, limit=limit, items=items)


@router.post("", response_model=ScenarioResponse, status_code=201)
async def create_scenario(
    body: ScenarioCreate,
    db: AsyncSession = Depends(get_db),
) -> ScenarioResponse:
    scenario = Scenario(
        name=body.name,
        description=body.description,
        category=body.category,
        turns_template=body.turns_template,
        user_persona=body.user_persona,
        constraints=body.constraints,
        difficulty=body.difficulty,
        tags=body.tags,
    )
    db.add(scenario)
    await db.flush()
    await db.refresh(scenario)
    return ScenarioResponse.model_validate(scenario)


@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(
    scenario_id: str,
    db: AsyncSession = Depends(get_db),
) -> ScenarioResponse:
    result = await db.execute(select(Scenario).where(Scenario.id == scenario_id))
    scenario = result.scalar_one_or_none()
    if not scenario:
        raise NotFoundError("Scenario", scenario_id)
    return ScenarioResponse.model_validate(scenario)


@router.put("/{scenario_id}", response_model=ScenarioResponse)
async def update_scenario(
    scenario_id: str,
    body: ScenarioUpdate,
    db: AsyncSession = Depends(get_db),
) -> ScenarioResponse:
    result = await db.execute(select(Scenario).where(Scenario.id == scenario_id))
    scenario = result.scalar_one_or_none()
    if not scenario:
        raise NotFoundError("Scenario", scenario_id)

    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(scenario, key, value)

    await db.flush()
    await db.refresh(scenario)
    return ScenarioResponse.model_validate(scenario)


@router.delete("/{scenario_id}", status_code=204)
async def delete_scenario(
    scenario_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(select(Scenario).where(Scenario.id == scenario_id))
    scenario = result.scalar_one_or_none()
    if not scenario:
        raise NotFoundError("Scenario", scenario_id)
    scenario.is_active = False
    await db.flush()
