
from sqlalchemy import literal
from sqlmodel import Session

from app import database, main
from app.models import Chunk, Document


def test_search_rejects_empty_query(client):
    response = client.get("/search", params={"q": ""})
    assert response.status_code == 422


def test_search_rejects_whitespace_only_query(client):
    response = client.get("/search", params={"q": "   "})
    assert response.status_code == 422


def test_search_rejects_missing_query(client):
    response = client.get("/search")
    assert response.status_code == 422


def test_search_rejects_zero_k(client):
    response = client.get("/search", params={"q": "Python", "k": 0})
    assert response.status_code == 422


def test_search_rejects_k_above_maximum(client):
    response = client.get("/search", params={"q": "Python", "k": 51})
    assert response.status_code == 422


def test_search_returns_empty_when_no_documents_are_ready(
    client,
    monkeypatch,
):
    def fail_if_called(_):
        raise AssertionError("Embedding should not run without ready documents.")

    monkeypatch.setattr(main, "generate_embedding", fail_if_called)

    response = client.get(
        "/search",
        params={"q": "What is this document about?"},
    )

    assert response.status_code == 200
    assert response.json() == []


def test_search_returns_only_chunks_from_ready_documents(
    client,
    monkeypatch,
):
    # Mock pgvector distance calculation for SQLite.
    vector_type = Chunk.__table__.c.embedding.type
    comparator_class = vector_type.comparator_factory


    monkeypatch.setattr(
        comparator_class,
        "cosine_distance",
        lambda self, embedding: literal(0.0),
    )

    monkeypatch.setattr(
        main,
        "generate_embedding",
        lambda question: [0.0] * 768,
    )

    statuses = ["ready", "pending", "processing", "failed"]

    with Session(database.engine) as session:
        for status in statuses:
            document = Document(
                filename=f"{status}.txt",
                content_type="text/plain",
                size_bytes=100,
                status=status,
            )

            session.add(document)
            session.flush()

            session.add(
                Chunk(
                    document_id=document.id,
                    text=f"Content from {status} document",
                    filename=document.filename,
                    page_number=1,
                    embedding=[0.0] * 768,
                )
            )

        session.commit()

    response = client.get(
        "/search",
        params={"q": "document content"},
    )

    assert response.status_code == 200

    results = response.json()

    assert len(results) == 1
    assert results[0]["filename"] == "ready.txt"
