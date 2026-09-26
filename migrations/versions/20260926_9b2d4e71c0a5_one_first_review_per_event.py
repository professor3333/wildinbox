"""one first review per event

The unique constraint on reviews.previous_review_id makes each review
superseded at most once, but NULLs never collide, so two concurrent first
reviews of an event could both be stored as roots of the history. A partial
unique index allows one review without a predecessor per event.

The upgrade refuses to run over histories that are already forked rather
than choosing which review to keep.

Revision ID: 9b2d4e71c0a5
Revises: 4803c5c84b95
Create Date: 2026-09-26 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9b2d4e71c0a5"
down_revision: str | None = "4803c5c84b95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    forked = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT event_id FROM reviews WHERE previous_review_id IS NULL "
                "GROUP BY event_id HAVING count(*) > 1 ORDER BY event_id"
            )
        )
        .scalars()
        .all()
    )
    if forked:
        raise RuntimeError(
            f"{len(forked)} event(s) already have more than one first review, e.g. "
            f"{[str(e) for e in forked[:5]]}; resolve them before upgrading"
        )
    op.create_index(
        "uq_reviews_one_first_review_per_event",
        "reviews",
        ["event_id"],
        unique=True,
        postgresql_where=sa.text("previous_review_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_reviews_one_first_review_per_event", table_name="reviews")
