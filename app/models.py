
from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    uploaded_at: datetime
