"""index events in list order

Every event list is ordered by (start_at, id). Without an index each page
sorted all matching events, about 60,000 for the review queue with a year of
history, to return 8.

Revision ID: c3a8f5e1d7b4
Revises: b7e1d4c9f2a0
Create Date: 2026-09-27 21:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c3a8f5e1d7b4"
down_revision: str | None = "b7e1d4c9f2a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_events_start_at_id", "events", ["start_at", "id"])


def downgrade() -> None:
    op.drop_index("ix_events_start_at_id", table_name="events")
