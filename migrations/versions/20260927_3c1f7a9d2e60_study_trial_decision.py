"""study trial decision

Records which decision's suggestion a participant saw in the "suggested"
condition, so the export states it rather than leaving it to be inferred.
Trials logged before this column existed keep NULL.

Revision ID: 3c1f7a9d2e60
Revises: 9b2d4e71c0a5
Create Date: 2026-09-27 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3c1f7a9d2e60"
down_revision: str | None = "9b2d4e71c0a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("study_trials", sa.Column("decision_id", sa.Uuid(), nullable=True))


def downgrade() -> None:
    op.drop_column("study_trials", "decision_id")
