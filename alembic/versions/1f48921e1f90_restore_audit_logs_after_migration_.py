"""restore audit logs if missing

Revision ID: 1f48921e1f90
Revises: 018091821793
Create Date: 2026-09-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "1f48921e1f90"
down_revision: Union[str, Sequence[str], None] = "018091821793"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Restore audit_logs only when it is missing."""

    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "audit_logs" not in inspector.get_table_names():
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), nullable=True),
            sa.Column("event_type", sa.String(length=50), nullable=False),
            sa.Column("success", sa.Boolean(), nullable=False),
            sa.Column("similarity", sa.String(length=20), nullable=True),
            sa.Column("details", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["user_id"],
                ["users.id"],
                name="audit_logs_user_id_fkey",
            ),
        )

        op.create_index(
            "ix_audit_logs_id",
            "audit_logs",
            ["id"],
            unique=False,
        )

        op.create_index(
            "ix_audit_logs_user_id",
            "audit_logs",
            ["user_id"],
            unique=False,
        )

        op.create_index(
            "ix_audit_logs_event_type",
            "audit_logs",
            ["event_type"],
            unique=False,
        )

        op.create_index(
            "ix_audit_logs_created_at",
            "audit_logs",
            ["created_at"],
            unique=False,
        )


def downgrade() -> None:
    """No-op downgrade.

    The audit_logs table may contain existing audit history.
    It must not be dropped automatically.
    """
    pass

    