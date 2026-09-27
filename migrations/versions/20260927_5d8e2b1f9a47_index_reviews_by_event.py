"""index reviews by event

Finding an event's current review (the `animal` filter on GET /events, and
review history) looks reviews up by event. The only index on
reviews.event_id is partial (first reviews only), so each lookup scanned the
whole table.

Revision ID: 5d8e2b1f9a47
Revises: 3c1f7a9d2e60
Create Date: 2026-09-27 12:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "5d8e2b1f9a47"
down_revision: str | None = "3c1f7a9d2e60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_reviews_event_id", "reviews", ["event_id"])


def downgrade() -> None:
    op.drop_index("ix_reviews_event_id", table_name="reviews")
