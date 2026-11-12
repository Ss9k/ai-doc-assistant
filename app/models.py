from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field as PydanticField
from sqlalchemy import Column, Numeric
from sqlmodel import Field, SQLModel


class Document(SQLModel, table=True):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
    )
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime


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


class ResumeData(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[str] = PydanticField(default_factory=list)
