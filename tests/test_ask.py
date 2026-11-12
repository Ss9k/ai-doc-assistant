import pytest
from sqlmodel import Session, select

from app import database
from app.main import GEMINI_MODEL
from app.models import LLMCall


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def successful_generate(prompt, provider, on_usage=None):
    if on_usage:
        on_usage(
            {
                "model": GEMINI_MODEL,
                "input_tokens": 100,
                "output_tokens": 50,
            }
        )

    yield "Hello "
    yield "World!"


async def failing_generate(prompt, provider, on_usage=None):
    raise RuntimeError("LLM provider failed")
    yield  # Makes this an async generator.


async def timeout_generate(prompt, provider, on_usage=None):
    raise TimeoutError("LLM request timed out")
    yield  # Makes this an async generator.


def test_ask_streams_response_and_saves_usage(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.generate",
        successful_generate,
    )

    response = client.post(
        "/ask",
        json={
            "question": "Explain Python",
            "provider": "gemini",
        },
    )

    assert response.status_code == 200
    assert response.text == "Hello World!"

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

    assert response.status_code == 400
    assert "Unsupported provider" in response.json()["detail"]


def test_ask_returns_503_when_llm_fails(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.generate",
        failing_generate,
    )

    response = client.post(
        "/ask",
        json={
            "question": "Hello",
            "provider": "gemini",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == ("LLM provider is currently unavailable.")


def test_ask_returns_504_on_timeout(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.generate",
        timeout_generate,
    )

    response = client.post(
        "/ask",
        json={
            "question": "Hello",
            "provider": "gemini",
        },
    )

    assert response.status_code == 504
    assert response.json()["detail"] == ("LLM request timed out after retries.")
