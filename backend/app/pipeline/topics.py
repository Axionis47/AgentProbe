"""Kafka topic name constants."""

CONVERSATION_COMPLETED = "agent.conversation.completed"
EVALUATION_SCORE_COMPLETED = "evaluation.score.completed"
METRICS_AGGREGATED = "metrics.aggregated"
PIPELINE_ERRORS = "pipeline.errors"

ALL_TOPICS = [
    CONVERSATION_COMPLETED,
    EVALUATION_SCORE_COMPLETED,
    METRICS_AGGREGATED,
    PIPELINE_ERRORS,
]
