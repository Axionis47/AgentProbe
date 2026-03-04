from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RubricCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    dimensions: list[dict[str, Any]] = Field(..., min_length=1)


class RubricUpdate(BaseModel):
    """Creates a new version of the rubric (rubrics are immutable once created)."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    dimensions: list[dict[str, Any]] | None = None


class RubricResponse(BaseModel):
    id: str
    name: str
    description: str | None
    dimensions: list[dict[str, Any]]
    version: int
    parent_id: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class RubricListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[RubricResponse]
