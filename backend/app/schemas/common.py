from enum import Enum

from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    total: int
    offset: int
    limit: int


class EvalRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING_SIMULATION = "running_simulation"
    RUNNING_EVALUATION = "running_evaluation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConversationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluatorType(str, Enum):
    MODEL_JUDGE = "model_judge"
    RUBRIC_GRADER = "rubric_grader"
    HUMAN = "human"
    REFERENCE_BASED = "reference_based"
    TRAJECTORY = "trajectory"
    PAIRWISE_JUDGE = "pairwise_judge"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"
