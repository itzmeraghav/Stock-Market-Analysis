"""AddedOptionTrading

Revision ID: 82ab9c268ba9
Revises: 2f2b2834b4e1
Create Date: 2026-08-29 06:51:58.751864

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "82ab9c268ba9"
down_revision: str | Sequence[str] | None = "2f2b2834b4e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "option_calculations",
        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "stock_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "calculation_date",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "spot_price",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "strike_price",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "time_to_expiry",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "risk_free_rate",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "volatility",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "call_price",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "put_price",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "call_delta",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "put_delta",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "gamma",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "vega",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "theta",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "rho",
            sa.Float(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["stock_id"],
            ["stocks.id"],
            ondelete="CASCADE",
        ),
    )

    op.create_index(
        "ix_option_calculations_stock_id",
        "option_calculations",
        ["stock_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(
        "ix_option_calculations_stock_id",
        table_name="option_calculations",
    )

    op.drop_table("option_calculations")
