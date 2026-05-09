from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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
    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata_")
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class ConversationListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[ConversationResponse]


class SimilarConversationItem(BaseModel):
    """One conversation in a similarity-search response.

    `similarity` is in [0, 1] where 1 is identical (cosine similarity flipped
    from Chroma's distance). `metadata` carries the per-evaluator scores that
    the evaluation_consumer backfills, so callers can sort/filter without a
    second hop to the evaluations table.
    """

    conversation: ConversationResponse
    similarity: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimilarConversationsResponse(BaseModel):
    source_conversation_id: str
    items: list[SimilarConversationItem]
