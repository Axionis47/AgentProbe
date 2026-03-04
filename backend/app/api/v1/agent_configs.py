from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.models.agent_config import AgentConfig
from app.schemas.agent_config import (
    AgentConfigCreate,
    AgentConfigListResponse,
    AgentConfigResponse,
    AgentConfigUpdate,
)

router = APIRouter(prefix="/agent-configs", tags=["agent-configs"])


@router.get("", response_model=AgentConfigListResponse)
async def list_agent_configs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    is_active: bool | None = None,
    model: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> AgentConfigListResponse:
    query = select(AgentConfig)
    count_query = select(func.count(AgentConfig.id))

    if is_active is not None:
        query = query.where(AgentConfig.is_active == is_active)
        count_query = count_query.where(AgentConfig.is_active == is_active)
    if model:
        query = query.where(AgentConfig.model == model)
        count_query = count_query.where(AgentConfig.model == model)

    total = (await db.execute(count_query)).scalar_one()
    result = await db.execute(
        query.order_by(AgentConfig.created_at.desc()).offset(offset).limit(limit)
    )
    items = [AgentConfigResponse.model_validate(r) for r in result.scalars().all()]

    return AgentConfigListResponse(total=total, offset=offset, limit=limit, items=items)


@router.post("", response_model=AgentConfigResponse, status_code=201)
async def create_agent_config(
    body: AgentConfigCreate,
    db: AsyncSession = Depends(get_db),
) -> AgentConfigResponse:
    agent_config = AgentConfig(
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        tools=body.tools,
        metadata_=body.metadata,
    )
    db.add(agent_config)
    await db.flush()
    await db.refresh(agent_config)
    return AgentConfigResponse.model_validate(agent_config)


@router.get("/{config_id}", response_model=AgentConfigResponse)
async def get_agent_config(
    config_id: str,
    db: AsyncSession = Depends(get_db),
) -> AgentConfigResponse:
    result = await db.execute(select(AgentConfig).where(AgentConfig.id == config_id))
    agent_config = result.scalar_one_or_none()
    if not agent_config:
        raise NotFoundError("AgentConfig", config_id)
    return AgentConfigResponse.model_validate(agent_config)


@router.put("/{config_id}", response_model=AgentConfigResponse)
async def update_agent_config(
    config_id: str,
    body: AgentConfigUpdate,
    db: AsyncSession = Depends(get_db),
) -> AgentConfigResponse:
    result = await db.execute(select(AgentConfig).where(AgentConfig.id == config_id))
    agent_config = result.scalar_one_or_none()
    if not agent_config:
        raise NotFoundError("AgentConfig", config_id)

    update_data = body.model_dump(exclude_unset=True)
    if "metadata" in update_data:
        update_data["metadata_"] = update_data.pop("metadata")
    for key, value in update_data.items():
        setattr(agent_config, key, value)

    await db.flush()
    await db.refresh(agent_config)
    return AgentConfigResponse.model_validate(agent_config)


@router.delete("/{config_id}", status_code=204)
async def delete_agent_config(
    config_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(select(AgentConfig).where(AgentConfig.id == config_id))
    agent_config = result.scalar_one_or_none()
    if not agent_config:
        raise NotFoundError("AgentConfig", config_id)
    agent_config.is_active = False
    await db.flush()
