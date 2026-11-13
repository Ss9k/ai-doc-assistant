from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from pydantic import BaseModel, Field as PydanticField
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Numeric
from sqlmodel import Field, SQLModel


class ResumeData(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[str] = PydanticField(default_factory=list)


class Document(SQLModel, table=True):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
    )
    filename: str
    content_type: str
    size_bytes: int
    storage_path: str | None = Field(default=None)
    status: str = Field(
        default="pending",
        index=True,
    )
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime
    status: str


class Chunk(SQLModel, table=True):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
    )

    document_id: str = Field(
        foreign_key="document.id",
        index=True,
    )

    text: str
    filename: str
    page_number: int

    embedding: list[float] = Field(
        sa_column=Column(
            Vector(768),
            nullable=False,
        ),
    )


class LLMCall(SQLModel, table=True):
    __tablename__ = "llm_call"

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
    )

    provider: str
    model: str

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    latency_ms: int = Field(ge=0)

    cost_usd: Decimal = Field(
        default=Decimal(0),
        sa_column=Column(
            Numeric(12, 8),
            nullable=False,
        ),
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
