"""worker processes

Revision ID: 3f1c2a9d7e40
Revises: 12c9bcb82636
Create Date: 2026-09-25 18:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3f1c2a9d7e40"
down_revision: str | None = "12c9bcb82636"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "worker_processes",
        sa.Column("id", sa.String(length=200), nullable=False),
        sa.Column("hostname", sa.String(length=200), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rss_bytes", sa.BigInteger(), nullable=True),
        sa.Column("peak_rss_bytes", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_worker_processes")),
    )


def downgrade() -> None:
    op.drop_table("worker_processes")
