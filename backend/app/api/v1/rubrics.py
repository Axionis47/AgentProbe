from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.models.rubric import Rubric
from app.schemas.rubric import (
    RubricCreate,
    RubricListResponse,
    RubricResponse,
    RubricUpdate,
)

router = APIRouter(prefix="/rubrics", tags=["rubrics"])


@router.get("", response_model=RubricListResponse)
async def list_rubrics(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    is_active: bool | None = None,
    db: AsyncSession = Depends(get_db),
) -> RubricListResponse:
    query = select(Rubric)
    count_query = select(func.count(Rubric.id))

    if is_active is not None:
        query = query.where(Rubric.is_active == is_active)
        count_query = count_query.where(Rubric.is_active == is_active)

    total = (await db.execute(count_query)).scalar_one()
    result = await db.execute(
        query.order_by(Rubric.created_at.desc()).offset(offset).limit(limit)
    )
    items = [RubricResponse.model_validate(r) for r in result.scalars().all()]

    return RubricListResponse(total=total, offset=offset, limit=limit, items=items)


@router.post("", response_model=RubricResponse, status_code=201)
async def create_rubric(
    body: RubricCreate,
    db: AsyncSession = Depends(get_db),
) -> RubricResponse:
    rubric = Rubric(
        name=body.name,
        description=body.description,
        dimensions=body.dimensions,
        version=1,
    )
    db.add(rubric)
    await db.flush()
    await db.refresh(rubric)
    return RubricResponse.model_validate(rubric)


@router.get("/{rubric_id}", response_model=RubricResponse)
async def get_rubric(
    rubric_id: str,
    db: AsyncSession = Depends(get_db),
) -> RubricResponse:
    result = await db.execute(select(Rubric).where(Rubric.id == rubric_id))
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise NotFoundError("Rubric", rubric_id)
    return RubricResponse.model_validate(rubric)


@router.put("/{rubric_id}", response_model=RubricResponse)
async def update_rubric(
    rubric_id: str,
    body: RubricUpdate,
    db: AsyncSession = Depends(get_db),
) -> RubricResponse:
    """Creates a new version of the rubric. Rubrics are immutable once created."""
    result = await db.execute(select(Rubric).where(Rubric.id == rubric_id))
    old_rubric = result.scalar_one_or_none()
    if not old_rubric:
        raise NotFoundError("Rubric", rubric_id)

    # Create new version
    new_rubric = Rubric(
        name=body.name or old_rubric.name,
        description=body.description if body.description is not None else old_rubric.description,
        dimensions=body.dimensions or old_rubric.dimensions,
        version=old_rubric.version + 1,
        parent_id=old_rubric.id,
    )
    db.add(new_rubric)
    await db.flush()
    await db.refresh(new_rubric)
    return RubricResponse.model_validate(new_rubric)


@router.get("/{rubric_id}/versions", response_model=list[RubricResponse])
async def list_rubric_versions(
    rubric_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[RubricResponse]:
    """List all versions of a rubric by tracing the parent chain."""
    result = await db.execute(select(Rubric).where(Rubric.id == rubric_id))
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise NotFoundError("Rubric", rubric_id)

    # Collect all versions with the same name
    result = await db.execute(
        select(Rubric)
        .where(Rubric.name == rubric.name)
        .order_by(Rubric.version.desc())
    )
    versions = [RubricResponse.model_validate(r) for r in result.scalars().all()]
    return versions


@router.delete("/{rubric_id}", status_code=204)
async def delete_rubric(
    rubric_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(select(Rubric).where(Rubric.id == rubric_id))
    rubric = result.scalar_one_or_none()
    if not rubric:
        raise NotFoundError("Rubric", rubric_id)
    rubric.is_active = False
    await db.flush()
