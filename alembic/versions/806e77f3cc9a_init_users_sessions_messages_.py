"""init users sessions messages investigations plans editions

Revision ID: 806e77f3cc9a
Revises: 
Create Date: 2026-08-30 19:20:11.943522

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '806e77f3cc9a'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("supabase_user_id", sa.UUID(), nullable=False, unique=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("role", sa.String(), server_default="user"),
        sa.Column("allow_aggregated_use", sa.Boolean(), server_default="false"),
        sa.Column("profile_hints", sa.JSON(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_supabase_user_id", "users", ["supabase_user_id"])

    op.create_table(
        "editions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("pdf_path", sa.String(), nullable=True),
        sa.Column("pages", sa.String(), nullable=True),
        sa.Column("chunks", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("session_id", sa.UUID(), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])

    op.create_table(
        "investigations",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("edition_id", sa.String(), sa.ForeignKey("editions.id"), nullable=True),
        sa.Column("query", sa.Text(), nullable=True),
        sa.Column("divisions", sa.JSON(), server_default="[]"),
        sa.Column("location", sa.JSON(), server_default="{}"),
        sa.Column("price_variants", sa.JSON(), server_default="[]"),
        sa.Column("client_hint", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_investigations_user_id", "investigations", ["user_id"])

    op.create_table(
        "plans",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("investigation_id", sa.UUID(), sa.ForeignKey("investigations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("edition_id", sa.String(), sa.ForeignKey("editions.id"), nullable=True),
        sa.Column("total_hectares", sa.String(), nullable=True),
        sa.Column("season", sa.String(), nullable=True),
        sa.Column("divisions", sa.JSON(), server_default="[]"),
        sa.Column("location", sa.JSON(), server_default="{}"),
        sa.Column("price_variants", sa.JSON(), server_default="[]"),
        sa.Column("client_hint", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_plans_user_id", "plans", ["user_id"])

    op.create_table(
        "pinned_insights",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.UUID(), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True),
        sa.Column("query_context", sa.JSON(), server_default="{}"),
        sa.Column("chart_payload", sa.JSON(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("pinned_insights")
    op.drop_table("plans")
    op.drop_table("investigations")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("editions")
    op.drop_table("users")
