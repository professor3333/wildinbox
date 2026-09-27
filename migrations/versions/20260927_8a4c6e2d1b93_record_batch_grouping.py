"""record each batch's grouping

Uploads may now choose the grouping interval, and the worker groups from the
interval recorded in the batch's manifest instead of the code's default.
Batches created before that were grouped with the default of the time, 5
seconds; this records it in their manifests, so a later change to the default
cannot regroup them.

Revision ID: 8a4c6e2d1b93
Revises: 5d8e2b1f9a47
Create Date: 2026-09-27 15:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8a4c6e2d1b93"
down_revision: str | None = "5d8e2b1f9a47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literal values, not imports: a migration must keep meaning what it meant
# when it was written. No sequence rule is claimed: uploads used
# sequence_id/v1 before v2, and each event records the rule that made it.
LEGACY = (
    '{"gap_seconds": 5.0, "gap_source": "legacy_default", "time_gap_rule": "time_gap/v1(gap_s=5)"}'
)


def upgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE batches SET manifest = manifest || jsonb_build_object("
            "'grouping', CAST(:legacy AS jsonb)) WHERE NOT manifest ? 'grouping'"
        ).bindparams(legacy=LEGACY)
    )


def downgrade() -> None:
    # Only what upgrade added; a gap an upload chose stays on record.
    op.execute(
        sa.text(
            "UPDATE batches SET manifest = manifest - 'grouping' "
            "WHERE manifest -> 'grouping' ->> 'gap_source' = 'legacy_default'"
        )
    )
