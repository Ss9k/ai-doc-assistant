from app import main


def test_search_rejects_empty_query(client):
    response = client.get("/search", params={"q": ""})

    assert response.status_code == 422


def test_search_returns_empty_when_no_documents_are_ready(
    client,
    monkeypatch,
):
    def fail_if_called(_):
        raise AssertionError("Embedding should not run without ready documents.")

    monkeypatch.setattr(main, "generate_embedding", fail_if_called)

    response = client.get("/search", params={"q": "What is this document about?"})

    assert response.status_code == 200
    assert response.json() == []
