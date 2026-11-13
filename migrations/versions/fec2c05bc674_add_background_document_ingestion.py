"""add background document ingestion

Revision ID: fec2c05bc674
Revises: b9ab0638a53b
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = "fec2c05bc674"
down_revision: Union[str, Sequence[str], None] = "b9ab0638a53b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector before creating a VECTOR column.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "document",
        sa.Column("storage_path", sa.String(), nullable=True),
    )

    # Existing documents receive the pending status.
    op.add_column(
        "document",
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default="pending",
        ),
    )

    op.create_index(
        "ix_document_status",
        "document",
        ["status"],
        unique=False,
    )

    op.create_table(
        "chunk",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["document.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_chunk_document_id",
        "chunk",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_document_id", table_name="chunk")
    op.drop_table("chunk")

    op.drop_index("ix_document_status", table_name="document")
    op.drop_column("document", "status")
    op.drop_column("document", "storage_path")

    # Intentionally keep the vector extension enabled.
