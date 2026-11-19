
import pytest
from sqlmodel import Session, select

from app import database
from app.main import GEMINI_MODEL
from app.models import Chunk, LLMCall


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_chunk(
    filename="python_guide.pdf",
    page_number=3,
    text="Python is a high-level programming language.",
):
    return Chunk(
        document_id="test-document-id",
        filename=filename,
        page_number=page_number,
        text=text,
        embedding=[0.0] * 768,
    )


def mock_retrieval(monkeypatch, chunks=None):
    if chunks is None:
        chunks = []

    monkeypatch.setattr(
        "app.main.retrieve_for_ask",
        lambda question: chunks,
    )


async def successful_generate(
    prompt,
    provider,
    on_usage=None,
    system_prompt=None,
):
    assert "DOCUMENT DATA:" in prompt
    assert "USER QUESTION:" in prompt
    assert "Python is a high-level programming language." in prompt
    assert "ONLY information explicitly supported" in prompt

    assert system_prompt is not None
    assert "untrusted document data" in system_prompt

    if on_usage:
        on_usage(
            {
                "model": GEMINI_MODEL,
                "input_tokens": 100,
                "output_tokens": 50,
            }
        )

    yield "Python is a high-level programming language [Source 1]."


async def failing_generate(
    prompt,
    provider,
    on_usage=None,
    system_prompt=None,
):
    raise RuntimeError("LLM provider failed")
    yield


async def timeout_generate(
    prompt,
    provider,
    on_usage=None,
    system_prompt=None,
):
    raise TimeoutError("LLM request timed out")
    yield


def test_ask_returns_grounded_response_and_saves_usage(
    client,
    monkeypatch,
):
    chunks = [
        (make_chunk(), 0.1),
    ]

    mock_retrieval(monkeypatch, chunks)

    monkeypatch.setattr(
        "app.main.generate",
        successful_generate,
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is Python?",
            "provider": "gemini",
        },
    )

    assert response.status_code == 200
    assert "Python is a high-level programming language" in response.text
    assert "Sources:" in response.text
    assert "python_guide.pdf" in response.text
    assert "Page 3" in response.text

    with Session(database.engine) as session:
        calls = session.exec(select(LLMCall)).all()

    assert len(calls) == 1

    call = calls[0]
    assert call.provider == "gemini"
    assert call.model == GEMINI_MODEL
    assert call.input_tokens == 100
    assert call.output_tokens == 50
    assert call.latency_ms >= 0
    assert call.cost_usd > 0


def test_ask_rejects_empty_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "   ",
            "provider": "gemini",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_ask_rejects_unsupported_provider(client):
    response = client.post(
        "/ask",
        json={
            "question": "Hello",
            "provider": "unsupported",
        },
    )

    assert response.status_code == 422

    detail = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "provider"]
        and error["type"] == "literal_error"
        for error in detail
    )


def test_ask_returns_i_dont_know_when_no_documents_are_ready(
    client,
    monkeypatch,
):
    mock_retrieval(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not run without document context.")

    monkeypatch.setattr("app.main.generate", fail_if_called)

    response = client.post(
        "/ask",
        json={
            "question": "What is Python?",
            "provider": "gemini",
        },
    )

    assert response.status_code == 200
    assert response.text == "I don't know."


def test_ask_returns_i_dont_know_when_no_relevant_chunks(
    client,
    monkeypatch,
):
    mock_retrieval(monkeypatch)

    response = client.post(
        "/ask",
        json={
            "question": "Explain an unrelated topic.",
            "provider": "gemini",
        },
    )

    assert response.status_code == 200
    assert response.text == "I don't know."


def test_ask_includes_multiple_source_references(
    client,
    monkeypatch,
):
    chunks = [
        (make_chunk("python.pdf", 2), 0.1),
        (make_chunk("python.pdf", 3), 0.2),
        (make_chunk("guide.pdf", 5), 0.3),
    ]

    mock_retrieval(monkeypatch, chunks)

    async def generate_with_sources(
        prompt,
        provider,
        on_usage=None,
        system_prompt=None,
    ):
        yield (
            "Answer based on the documents "
            "[Source 1] [Source 2] [Source 3]."
        )

    monkeypatch.setattr(
        "app.main.generate",
        generate_with_sources,
    )

    response = client.post(
        "/ask",
        json={
            "question": "Explain Python.",
            "provider": "gemini",
        },
    )

    assert response.status_code == 200
    assert "[Source 1] python.pdf (Page 2)" in response.text
    assert "[Source 2] python.pdf (Page 3)" in response.text
    assert "[Source 3] guide.pdf (Page 5)" in response.text


def test_ask_returns_503_when_llm_fails(client, monkeypatch):
    mock_retrieval(
        monkeypatch,
        [(make_chunk(), 0.1)],
    )

    monkeypatch.setattr(
        "app.main.generate",
        failing_generate,
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is Python?",
            "provider": "gemini",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "LLM provider is currently unavailable."
    )


def test_ask_returns_504_on_timeout(client, monkeypatch):
    mock_retrieval(
        monkeypatch,
        [(make_chunk(), 0.1)],
    )

    monkeypatch.setattr(
        "app.main.generate",
        timeout_generate,
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is Python?",
            "provider": "gemini",
        },
    )

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "LLM request timed out after retries."
    )


def test_ask_does_not_show_sources_when_answer_is_unknown(
    client,
    monkeypatch,
):
    mock_retrieval(
        monkeypatch,
        [(make_chunk(), 0.1)],
    )

    async def generate_unknown(
        prompt,
        provider,
        on_usage=None,
        system_prompt=None,
    ):
        yield "I don't know."

    monkeypatch.setattr(
        "app.main.generate",
        generate_unknown,
    )

    response = client.post(
        "/ask",
        json={
            "question": "What is Python?",
            "provider": "gemini",
        },
    )

    assert response.status_code == 200
    assert response.text.strip() == "I don't know."
    assert "Sources:" not in response.text
