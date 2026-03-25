from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MetricResponse(BaseModel):
    id: str
    conversation_id: str
    metric_name: str
    value: float
    unit: str | None
    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata_")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class MetricListResponse(BaseModel):
    total: int
    items: list[MetricResponse]


class AggregatedMetricResponse(BaseModel):
    metric_name: str
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    sample_count: int
