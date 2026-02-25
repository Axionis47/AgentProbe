"""Initial schema with all 8 tables.

Revision ID: 001
Revises: 
Create Date: 2026-02-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(30), server_default="evaluator"),
        sa.Column("api_key_hash", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    # Agent configs table
    op.create_table(
        "agent_configs",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("model", sa.String(100), nullable=False, server_default="claude-sonnet-4-20250514"),
        sa.Column("temperature", sa.Float(), server_default="0.7"),
        sa.Column("max_tokens", sa.Integer(), server_default="4096"),
        sa.Column("tools", postgresql.JSONB(), server_default="[]"),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_agent_configs"),
    )

    # Scenarios table
    op.create_table(
        "scenarios",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("turns_template", postgresql.JSONB(), nullable=False),
        sa.Column("user_persona", postgresql.JSONB(), server_default="{}"),
        sa.Column("constraints", postgresql.JSONB(), server_default="{}"),
        sa.Column("difficulty", sa.String(20), server_default="medium"),
        sa.Column("tags", postgresql.ARRAY(sa.String()), server_default="{}"),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_scenarios"),
    )

    # Rubrics table
    op.create_table(
        "rubrics",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("dimensions", postgresql.JSONB(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("parent_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_rubrics"),
        sa.ForeignKeyConstraint(["parent_id"], ["rubrics.id"], name="fk_rubrics_parent_id_rubrics"),
    )
    op.create_index("idx_rubrics_name_version", "rubrics", ["name", sa.text("version DESC")])

    # Eval runs table
    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("agent_config_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("scenario_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("rubric_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("num_conversations", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("config", postgresql.JSONB(), server_default="{}"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_by", postgresql.UUID(as_uuid=False), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_eval_runs"),
        sa.ForeignKeyConstraint(["agent_config_id"], ["agent_configs.id"], name="fk_eval_runs_agent_config_id_agent_configs", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["scenario_id"], ["scenarios.id"], name="fk_eval_runs_scenario_id_scenarios", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["rubric_id"], ["rubrics.id"], name="fk_eval_runs_rubric_id_rubrics"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_eval_runs_created_by_users"),
    )
    op.create_index("idx_eval_runs_status", "eval_runs", ["status"])
    op.create_index("idx_eval_runs_agent_config", "eval_runs", ["agent_config_id"])
    op.create_index("idx_eval_runs_created", "eval_runs", [sa.text("created_at DESC")])

    # Conversations table
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("eval_run_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("sequence_num", sa.Integer(), nullable=False),
        sa.Column("turns", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), server_default="0"),
        sa.Column("total_input_tokens", sa.Integer(), server_default="0"),
        sa.Column("total_output_tokens", sa.Integer(), server_default="0"),
        sa.Column("total_latency_ms", sa.Integer(), server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
        sa.ForeignKeyConstraint(["eval_run_id"], ["eval_runs.id"], name="fk_conversations_eval_run_id_eval_runs", ondelete="CASCADE"),
    )
    op.create_index("idx_conversations_eval_run", "conversations", ["eval_run_id"])
    op.create_index("idx_conversations_status", "conversations", ["status"])

    # Evaluations table
    op.create_table(
        "evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("evaluator_type", sa.String(30), nullable=False),
        sa.Column("evaluator_id", sa.String(255), nullable=True),
        sa.Column("rubric_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("scores", postgresql.JSONB(), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("per_turn_scores", postgresql.JSONB(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_evaluations"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], name="fk_evaluations_conversation_id_conversations", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["rubric_id"], ["rubrics.id"], name="fk_evaluations_rubric_id_rubrics"),
    )
    op.create_index("idx_evaluations_conversation", "evaluations", ["conversation_id"])
    op.create_index("idx_evaluations_type", "evaluations", ["evaluator_type"])
    op.create_index("idx_evaluations_overall", "evaluations", ["overall_score"])

    # Metrics table
    op.create_table(
        "metrics",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(50), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_metrics"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], name="fk_metrics_conversation_id_conversations", ondelete="CASCADE"),
        sa.UniqueConstraint("conversation_id", "metric_name", name="uq_metrics_conv_name"),
    )
    op.create_index("idx_metrics_conversation", "metrics", ["conversation_id"])
    op.create_index("idx_metrics_name", "metrics", ["metric_name"])


def downgrade() -> None:
    op.drop_table("metrics")
    op.drop_table("evaluations")
    op.drop_table("conversations")
    op.drop_table("eval_runs")
    op.drop_table("rubrics")
    op.drop_table("scenarios")
    op.drop_table("agent_configs")
    op.drop_table("users")
