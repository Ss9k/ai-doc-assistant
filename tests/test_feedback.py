from sqlmodel import Session, select

from app import database
from app.main import langfuse
from app.models import Feedback


def test_submit_feedback_success(client, monkeypatch):
    score_calls = []

    def mock_create_score(**kwargs):
        score_calls.append(kwargs)

    monkeypatch.setattr(langfuse, "create_score", mock_create_score)
    monkeypatch.setattr("app.main.flush_langfuse", lambda: None)

    payload = {
        "trace_id": "test-trace-123",
        "rating": "thumbs_up",
    }

    response = client.post("/feedback", json=payload)

    assert response.status_code == 201
    assert response.json()["trace_id"] == payload["trace_id"]
    assert response.json()["rating"] == payload["rating"]

    with Session(database.engine) as session:
        feedback = session.exec(
            select(Feedback).where(Feedback.trace_id == payload["trace_id"])
        ).first()

    assert feedback is not None
    assert feedback.rating == "thumbs_up"

    assert len(score_calls) == 1
    assert score_calls[0]["name"] == "user-feedback"
    assert score_calls[0]["value"] == "thumbs_up"
    assert score_calls[0]["trace_id"] == "test-trace-123"
    assert score_calls[0]["data_type"] == "CATEGORICAL"


def test_submit_thumbs_down(client, monkeypatch):
    monkeypatch.setattr(langfuse, "create_score", lambda **kwargs: None)
    monkeypatch.setattr("app.main.flush_langfuse", lambda: None)

    response = client.post(
        "/feedback",
        json={
            "trace_id": "test-trace-456",
            "rating": "thumbs_down",
        },
    )

    assert response.status_code == 201
    assert response.json()["rating"] == "thumbs_down"


def test_submit_feedback_invalid_rating(client):
    response = client.post(
        "/feedback",
        json={
            "trace_id": "test-trace-789",
            "rating": "neutral",
        },
    )

    assert response.status_code == 422


def test_submit_feedback_empty_trace_id(client):
    response = client.post(
        "/feedback",
        json={
            "trace_id": "",
            "rating": "thumbs_up",
        },
    )

    assert response.status_code == 422


def test_submit_duplicate_feedback(client, monkeypatch):
    monkeypatch.setattr(langfuse, "create_score", lambda **kwargs: None)
    monkeypatch.setattr("app.main.flush_langfuse", lambda: None)

    payload = {
        "trace_id": "duplicate-trace-123",
        "rating": "thumbs_up",
    }

    first_response = client.post("/feedback", json=payload)
    second_response = client.post("/feedback", json=payload)

    assert first_response.status_code == 201
    assert second_response.status_code == 409
