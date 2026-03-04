"""Create graphs table for graph snapshots.

Revision ID: 003
Revises: 002
Create Date: 2026-03-03
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graphs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("node_count", sa.Integer(), nullable=False),
        sa.Column("cluster_count", sa.Integer(), server_default="0"),
        sa.Column("memory_count", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("metadata", JSONB(), server_default=sa.text("'{}'")),
    )
    op.create_index("idx_graphs_namespace", "graphs", ["namespace"])
    op.create_index("idx_graphs_namespace_latest", "graphs", ["namespace", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_graphs_namespace_latest", table_name="graphs")
    op.drop_index("idx_graphs_namespace", table_name="graphs")
    op.drop_table("graphs")
