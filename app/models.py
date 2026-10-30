from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel
from sqlmodel import SQLModel, Field


class Document(SQLModel, table=True):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        primary_key=True,
    )
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime
