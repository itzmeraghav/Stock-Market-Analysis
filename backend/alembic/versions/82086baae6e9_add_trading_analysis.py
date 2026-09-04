"""Add trading analysis

Revision ID: 82086baae6e9
Revises: 82ab9c268ba9
Create Date: 2026-08-29 18:06:45.735412

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "82086baae6e9"
down_revision: str | Sequence[str] | None = "82ab9c268ba9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create trading_analyses table."""

    op.create_table(
        "trading_analyses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stock_id", sa.Integer(), nullable=False),
        sa.Column(
            "calculation_date",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("current_price", sa.Float(), nullable=False),
        sa.Column("predicted_price", sa.Float(), nullable=False),
        sa.Column(
            "expected_change_percent",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "direction",
            sa.String(length=10),
            nullable=False,
        ),
        sa.Column(
            "model_name",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "investment_amount",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "shares",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "invested_amount",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "remaining_amount",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "risk_percentage",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "stop_loss_price",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "target_percentage",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "target_price",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "maximum_loss",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "potential_profit",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "risk_reward_ratio",
            sa.Float(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["stock_id"],
            ["stocks.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_trading_analyses_stock_id",
        "trading_analyses",
        ["stock_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop trading_analyses table."""

    op.drop_index(
        "ix_trading_analyses_stock_id",
        table_name="trading_analyses",
    )

    op.drop_table("trading_analyses")
