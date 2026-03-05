"""Add 6 per-region embedding columns for multi-axis retrieval.

Revision ID: 004
Revises: 003
Create Date: 2026-03-05
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

REGIONS = ["hippo", "amyg", "pfc", "sensory", "striatum", "cerebellum"]
EMBEDDING_DIM = 1024


def upgrade() -> None:
    for region in REGIONS:
        col_name = f"{region}_embedding"
        op.add_column("memories", sa.Column(col_name, Vector(EMBEDDING_DIM), nullable=True))

    # Create ivfflat indexes for each region column.
    # In production, run with CONCURRENTLY during low-traffic window.
    for region in REGIONS:
        col_name = f"{region}_embedding"
        idx_name = f"idx_{col_name}"
        op.execute(
            f"CREATE INDEX {idx_name} "
            f"ON memories USING ivfflat ({col_name} vector_cosine_ops)"
        )


def downgrade() -> None:
    for region in REGIONS:
        col_name = f"{region}_embedding"
        idx_name = f"idx_{col_name}"
        op.drop_index(idx_name, table_name="memories")
        op.drop_column("memories", col_name)
