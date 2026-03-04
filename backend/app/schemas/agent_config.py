from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentConfigCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    system_prompt: str = Field(..., min_length=1)
    model: str = "claude-sonnet-4-20250514"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200000)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentConfigUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    system_prompt: str | None = Field(default=None, min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=200000)
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    is_active: bool | None = None


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    description: str | None
    system_prompt: str
    model: str
    temperature: float
    max_tokens: int
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AgentConfigListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[AgentConfigResponse]
