from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvalRunCreate(BaseModel):
    name: str | None = None
    agent_config_id: str
    scenario_id: str
    rubric_id: str | None = None
    num_conversations: int = Field(default=5, ge=1, le=100)
    config: dict[str, Any] = Field(default_factory=dict)


class EvalRunResponse(BaseModel):
    id: str
    name: str | None
    agent_config_id: str
    scenario_id: str
    rubric_id: str | None
    status: str
    num_conversations: int
    config: dict[str, Any]
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalRunListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[EvalRunResponse]
