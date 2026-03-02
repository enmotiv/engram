"""Resize embedding columns from 384 to 1024 dimensions for BGE-M3.

Revision ID: 002
Revises: 001
Create Date: 2026-03-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop and recreate embedding columns with new dimensions.
    # Existing embeddings are invalidated (different model) so data loss is expected.
    op.drop_column("memories", "embedding")
    op.add_column("memories", sa.Column("embedding", Vector(1024)))

    op.drop_column("edges", "context_embedding")
    op.add_column("edges", sa.Column("context_embedding", Vector(1024)))


def downgrade() -> None:
    op.drop_column("edges", "context_embedding")
    op.add_column("edges", sa.Column("context_embedding", Vector(384)))

    op.drop_column("memories", "embedding")
    op.add_column("memories", sa.Column("embedding", Vector(384)))
