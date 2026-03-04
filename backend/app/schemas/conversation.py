from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ConversationResponse(BaseModel):
    id: str
    eval_run_id: str
    sequence_num: int
    turns: list[dict[str, Any]]
    turn_count: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: int
    status: str
    error_message: str | None
    metadata: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[ConversationResponse]
