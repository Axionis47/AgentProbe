from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ScenarioCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    category: str | None = None
    turns_template: list[dict[str, Any]] = Field(..., min_length=1)
    user_persona: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)


class ScenarioUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    category: str | None = None
    turns_template: list[dict[str, Any]] | None = None
    user_persona: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None
    difficulty: str | None = None
    tags: list[str] | None = None
    is_active: bool | None = None


class ScenarioResponse(BaseModel):
    id: str
    name: str
    description: str | None
    category: str | None
    turns_template: list[dict[str, Any]]
    user_persona: dict[str, Any]
    constraints: dict[str, Any]
    difficulty: str
    tags: list[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ScenarioListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[ScenarioResponse]
