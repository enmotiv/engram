"""Create memories and edges tables.

Revision ID: 001
Revises: None
Create Date: 2026-02-28
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "memories",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("memory_type", sa.Text(), server_default="episodic"),
        sa.Column("embedding", Vector(384)),
        sa.Column("dimension_vectors", sa.JSON(), server_default="{}"),
        sa.Column("dimension_scores", sa.JSON(), server_default="{}"),
        sa.Column("features", sa.JSON(), server_default="{}"),
        sa.Column("feature_vector", Vector(32)),
        sa.Column("activation", sa.Float(), server_default="0.0"),
        sa.Column("salience", sa.Float(), server_default="0.5"),
        sa.Column("access_count", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_accessed", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_index("idx_memories_namespace", "memories", ["namespace"])

    op.create_table(
        "edges",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_memory_id", sa.UUID(), sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_memory_id", sa.UUID(), sa.ForeignKey("memories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("edge_type", sa.Text(), nullable=False),
        sa.Column("weight", sa.Float(), server_default="0.5"),
        sa.Column("context", sa.JSON(), server_default="{}"),
        sa.Column("context_features", sa.JSON()),
        sa.Column("context_embedding", Vector(384)),
        sa.Column("namespace", sa.Text(), server_default="'default'"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_activated", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("source_memory_id", "target_memory_id", "edge_type", name="uq_edge_source_target_type"),
    )

    op.create_index("idx_edges_source", "edges", ["source_memory_id"])
    op.create_index("idx_edges_target", "edges", ["target_memory_id"])
    op.create_index("idx_edges_source_type", "edges", ["source_memory_id", "edge_type"])


def downgrade() -> None:
    op.drop_table("edges")
    op.drop_table("memories")
