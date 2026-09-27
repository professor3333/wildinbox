"""index created_at for monitoring

Monitoring counts images and predictions created in its time window and
analyses events created in its history window. Without an index each count
scanned every row ever stored (about 0.3 s each at 250,000 images).

Revision ID: b7e1d4c9f2a0
Revises: 8a4c6e2d1b93
Create Date: 2026-09-27 18:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b7e1d4c9f2a0"
down_revision: str | None = "8a4c6e2d1b93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEXES = (
    ("ix_images_created_at", "images"),
    ("ix_predictions_created_at", "predictions"),
    ("ix_events_created_at", "events"),
)


def upgrade() -> None:
    for name, table in INDEXES:
        op.create_index(name, table, ["created_at"])


def downgrade() -> None:
    for name, table in INDEXES:
        op.drop_index(name, table_name=table)
