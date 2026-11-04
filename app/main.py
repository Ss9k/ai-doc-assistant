from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlmodel import Session, select

from app.database import get_session
from app.models import Document, DocumentResponse

app = FastAPI(title="AI Document Assistant")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=201,
)
async def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Allowed types: PDF, TXT, DOCX.",
        )

    content = await file.read()

    if not content or not content.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty or contains only whitespace.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 10 MB.",
        )

    if extension == ".pdf":
        try:
            pdf = PdfReader(BytesIO(content))
            has_text = any((page.extract_text() or "").strip() for page in pdf.pages)
        except (PdfReadError, EOFError):
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is invalid or unreadable.",
            )

        if not has_text:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF contains no extractable text.",
            )
    existing_document = session.exec(
        select(Document).where(Document.filename == filename)
    ).first()

    if existing_document:
        raise HTTPException(
            status_code=409,
            detail="A document with this filename already exists.",
        )

    document = Document(
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        uploaded_at=datetime.now(UTC),
    )

    session.add(document)
    session.commit()
    session.refresh(document)

    return document


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(session: Session = Depends(get_session)):
    statement = select(Document).order_by(Document.uploaded_at.desc())
    return session.exec(statement).all()
