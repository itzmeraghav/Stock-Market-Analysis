"""InitialCreate

Revision ID: 2f2b2834b4e1
Revises:
Create Date: 2026-08-23 20:02:27.942160

"""

from collections.abc import Sequence

revision: str = "2f2b2834b4e1"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
