"""reviews record the submitting principal

The authenticated principal that submitted each review: the reviewer, or an
authorized delegate recording on the reviewer's behalf. Existing rows stay
null (they were recorded before identities were enforced).

Revision ID: 4803c5c84b95
Revises: 3f1c2a9d7e40
Create Date: 2026-09-25 22:56:39.357098
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "4803c5c84b95"
down_revision: str | None = "3f1c2a9d7e40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("reviews", sa.Column("recorded_by", sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column("reviews", "recorded_by")
