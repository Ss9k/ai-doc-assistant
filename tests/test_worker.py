from sqlmodel import Session, select

from app import database
from app.main import process_document
from app.models import Chunk, Document


def create_text_document(tmp_path):
    file_path = tmp_path / "research.txt"
    file_path.write_text("Research content. " * 100, encoding="utf-8")

    document = Document(
        filename="research.txt",
        content_type="text/plain",
        size_bytes=file_path.stat().st_size,
        storage_path=str(file_path),
    )

    with Session(database.engine) as session:
        session.add(document)
        session.commit()
        session.refresh(document)
        document_id = document.id

    return document_id


def test_worker_marks_document_ready_and_prevents_duplicates(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "app.main.generate_embedding",
        lambda text: [0.1] * 768,
    )

    document_id = create_text_document(tmp_path)

    process_document(document_id)

    with Session(database.engine) as session:
        document = session.get(Document, document_id)
        chunks = session.exec(
            select(Chunk).where(Chunk.document_id == document_id)
        ).all()

        assert document.status == "ready"
        assert len(chunks) > 0

        first_count = len(chunks)

    # Reprocessing must replace old chunks, not duplicate them.
    process_document(document_id)

    with Session(database.engine) as session:
        document = session.get(Document, document_id)
        chunks = session.exec(
            select(Chunk).where(Chunk.document_id == document_id)
        ).all()

        assert document.status == "ready"
        assert len(chunks) == first_count


def test_worker_marks_document_failed_on_embedding_error(client, tmp_path, monkeypatch):
    def fail_embedding(text):
        raise RuntimeError("Embedding service unavailable")

    monkeypatch.setattr("app.main.generate_embedding", fail_embedding)

    document_id = create_text_document(tmp_path)

    process_document(document_id)

    with Session(database.engine) as session:
        document = session.get(Document, document_id)
        chunks = session.exec(
            select(Chunk).where(Chunk.document_id == document_id)
        ).all()

        assert document.status == "failed"
        assert len(chunks) == 0


def test_worker_marks_document_failed_when_file_is_missing(client, tmp_path):
    document = Document(
        filename="missing.txt",
        content_type="text/plain",
        size_bytes=10,
        storage_path=str(tmp_path / "missing.txt"),
    )

    with Session(database.engine) as session:
        session.add(document)
        session.commit()
        session.refresh(document)
        document_id = document.id

    process_document(document_id)

    with Session(database.engine) as session:
        document = session.get(Document, document_id)
        assert document.status == "failed"
