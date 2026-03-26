"""Add external agent support columns to agent_configs.

Revision ID: 002
Revises: 001
Create Date: 2026-03-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_configs",
        sa.Column("agent_type", sa.String(20), nullable=False, server_default="builtin"),
    )
    op.add_column(
        "agent_configs",
        sa.Column("endpoint_url", sa.String(500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_configs", "endpoint_url")
    op.drop_column("agent_configs", "agent_type")
