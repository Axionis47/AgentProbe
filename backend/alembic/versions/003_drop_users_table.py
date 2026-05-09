"""Drop the unused users table and the eval_runs.created_by FK.

Nothing ever wrote to either, so this is a straight removal — no
data migration needed. If we ever introduce real auth later it
should be a deliberate, separate piece of work, not piggy-backed
on whatever this stub was.

Revision ID: 003
Revises: 002
Create Date: 2026-05-09

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("fk_eval_runs_created_by_users", "eval_runs", type_="foreignkey")
    op.drop_column("eval_runs", "created_by")
    op.drop_table("users")


def downgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(30), server_default="evaluator"),
        sa.Column("api_key_hash", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.add_column(
        "eval_runs",
        sa.Column("created_by", postgresql.UUID(as_uuid=False), nullable=True),
    )
    op.create_foreign_key(
        "fk_eval_runs_created_by_users",
        "eval_runs",
        "users",
        ["created_by"],
        ["id"],
    )
