"""single-reviewer study design

Review study 2 (configs/study/review_study_2.yaml): a plan may carry a
design (session schedule and the suggestions to show), and trials may use the
"assisted" condition. Study 1 plans have no design and are unchanged.

Revision ID: e4b9c2a7d1f6
Revises: b7e1d4c9f2a0
Create Date: 2026-09-28 14:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e4b9c2a7d1f6"
down_revision: str | None = "b7e1d4c9f2a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "study_plans",
        sa.Column("design", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.drop_constraint(op.f("ck_study_trials_condition"), "study_trials", type_="check")
    op.create_check_constraint(
        op.f("ck_study_trials_condition"),
        "study_trials",
        "condition IN ('grouped', 'suggested', 'assisted')",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_study_trials_condition"), "study_trials", type_="check")
    op.create_check_constraint(
        op.f("ck_study_trials_condition"),
        "study_trials",
        "condition IN ('grouped', 'suggested')",
    )
    op.drop_column("study_plans", "design")
