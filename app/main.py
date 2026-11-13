import asyncio
import logging
import os
from uuid import uuid4
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlmodel import Session, select
from typing import Literal
from starlette.concurrency import run_in_threadpool

from app import database
from app.database import get_session
from app.llm import (
    GEMINI_MODEL,
    OLLAMA_MODEL,
    extract_resume,
    generate,
)
from app.models import Document, DocumentResponse, LLMCall, ResumeData
from app.pricing import calculate_cost

from app.llm import generate_embedding
from app.models import Chunk
from app.ingestion import chunk_text, extract_pdf_pages

app = FastAPI(title="AI Document Assistant")
logger = logging.getLogger(__name__)


def process_document(document_id: str) -> None:
    with Session(database.engine) as session:
        document = session.get(Document, document_id)

        if document is None:
            logger.error(
                "Document not found for background processing: %s",
                document_id,
            )
            return

        try:
            # 1. Mark document as processing.
            document.status = "processing"
            session.add(document)
            session.commit()

            logger.info(
                "Started processing document: %s",
                document.filename,
            )

            # 2. Validate stored file.
            if not document.storage_path:
                raise ValueError("Document storage path is missing.")

            pdf_path = Path(document.storage_path)

            if not pdf_path.is_file():
                raise FileNotFoundError(
                    f"Uploaded file not found: {pdf_path}"
                )

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(
                    "Background ingestion currently supports PDF files only."
                )

            # 3. Extract text page by page.
            pages = extract_pdf_pages(str(pdf_path))

            if not pages:
                raise ValueError(
                    "PDF contains no extractable text."
                )

            # 4. Generate chunks and embeddings.
            chunks_to_save = []

            for page_number, page_text in pages:
                text_chunks = chunk_text(
                    page_text,
                    chunk_size=800,
                    overlap=100,
                )

                for text in text_chunks:
                    embedding = generate_embedding(text)

                    chunks_to_save.append(
                        Chunk(
                            document_id=document.id,
                            text=text,
                            filename=document.filename,
                            page_number=page_number,
                            embedding=embedding,
                        )
                    )

            if not chunks_to_save:
                raise ValueError("No chunks were generated.")

            # 5. Save all chunks in one transaction.
            session.add_all(chunks_to_save)
            session.flush()

            # 6. Mark ready only after successful storage.
            document.status = "ready"
            session.add(document)
            session.commit()

            logger.info(
                "Successfully processed document %s: %d chunks saved.",
                document.filename,
                len(chunks_to_save),
            )

        except Exception:
            # Roll back any partially inserted chunks.
            session.rollback()

            try:
                document = session.get(Document, document_id)

                if document is not None:
                    document.status = "failed"
                    session.add(document)
                    session.commit()

            except Exception:
                session.rollback()
                logger.exception(
                    "Failed to update document status to failed: %s",
                    document_id,
                )

            logger.exception(
                "Background processing failed for document ID: %s",
                document_id,
            )

class ResumeExtractRequest(BaseModel):
    resume_text: str


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=201,
)
async def upload_document(
    background_tasks: BackgroundTasks,
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

    existing_document = session.exec(
        select(Document).where(Document.filename == filename)
    ).first()

    if existing_document:
        raise HTTPException(
            status_code=409,
            detail="A document with this filename already exists.",
        )

    # Generate a unique filename to prevent collisions.
    stored_filename = f"{uuid4().hex}{extension}"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = UPLOAD_DIR / stored_filename

    try:
        await run_in_threadpool(storage_path.write_bytes, content)

        document = Document(
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(content),
            storage_path=str(storage_path),
            uploaded_at=datetime.now(UTC),
        )

        session.add(document)
        session.commit()
        session.refresh(document)

        background_tasks.add_task(process_document, document.id)
        
        return document

    except Exception:
        session.rollback()

        if storage_path.exists():
            try:
                storage_path.unlink()
            except OSError:
                logger.exception("Failed to remove uploaded file: %s", storage_path)

        logger.exception("Failed to save uploaded document")
        raise HTTPException(
            status_code=500,
            detail="Failed to save the uploaded document.",
        )

    


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(session: Session = Depends(get_session)):
    statement = select(Document).order_by(Document.uploaded_at.desc())
    return session.exec(statement).all()


class AskRequest(BaseModel):
    question: str
    provider: Literal["gemini", "ollama"] = "gemini"


@app.post("/ask")
async def ask(request: AskRequest):
    question = request.question.strip()
    provider = request.provider.lower().strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    started_at = time.perf_counter()

    usage = {
        "model": GEMINI_MODEL if provider == "gemini" else OLLAMA_MODEL,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    logged = False

    def capture_usage(data):
        usage.update(data)

    def save_call():
        nonlocal logged

        if logged:
            return

        logged = True

        try:
            latency_ms = int((time.perf_counter() - started_at) * 1000)

            cost = calculate_cost(
                provider=provider,
                model=usage["model"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )

            call = LLMCall(
                provider=provider,
                model=usage["model"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                latency_ms=latency_ms,
                cost_usd=cost,
            )

            with Session(database.engine) as session:
                session.add(call)
                session.commit()

        except Exception:
            logger.exception("Failed to save LLM call")

    iterator = generate(
        question,
        provider,
        on_usage=capture_usage,
    )

    try:
        first_chunk = await anext(iterator)

    except StopAsyncIteration:
        first_chunk = None

    except (TimeoutError, asyncio.TimeoutError) as exc:
        await run_in_threadpool(save_call)
        raise HTTPException(
            status_code=504,
            detail="LLM request timed out after retries.",
        ) from exc

    except Exception as exc:
        logger.exception("LLM generation failed")
        await run_in_threadpool(save_call)
        raise HTTPException(
            status_code=503,
            detail="LLM provider is currently unavailable.",
        ) from exc

    async def response_stream() -> AsyncIterator[str]:
        try:
            if first_chunk is not None:
                yield first_chunk

            async for chunk in iterator:
                yield chunk

        except Exception:
            logger.exception("LLM stream interrupted")
            raise

        finally:
            await run_in_threadpool(save_call)

    return StreamingResponse(
        response_stream(),
        media_type="text/plain",
    )


@app.post("/extract", response_model=ResumeData)
async def extract_resume_endpoint(request: ResumeExtractRequest):
    if not request.resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Resume text cannot be empty.",
        )

    try:
        return await extract_resume(request.resume_text)

    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not extract valid resume data after two attempts.",
        ) from exc

    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Resume extraction timed out.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="Resume extraction service is not configured correctly.",
        ) from exc

    except Exception as exc:
        logging.getLogger(__name__).exception("Resume extraction failed")
        raise HTTPException(
            status_code=503,
            detail="Resume extraction service is currently unavailable.",
        ) from exc
