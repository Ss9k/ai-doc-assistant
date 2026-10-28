
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from io import BytesIO
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from app.models import DocumentResponse

app = FastAPI(title="AI Document Assistant")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# In-memory storage; data disappears when the server restarts.
documents: list[DocumentResponse] = []


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=201,
)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Allowed types: PDF, TXT, DOCX.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 10 MB.",
        )

    if extension == ".pdf":
        try:
            pdf = PdfReader(BytesIO(content))
            has_text = any(
                (page.extract_text() or "").strip()
                for page in pdf.pages
            )
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

    document = DocumentResponse(
        id=str(uuid4()),
        filename=filename,
        size_bytes=len(content),
        uploaded_at=datetime.now(timezone.utc),
    )

    documents.append(document)
    return document


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents():
    return documents
