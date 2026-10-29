
from io import BytesIO

from pypdf import PdfWriter


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_valid_document(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "research.txt",
                b"sample research content",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["filename"] == "research.txt"
    assert data["content_type"] == "text/plain"
    assert data["size_bytes"] == len(b"sample research content")
    assert "id" in data
    assert "uploaded_at" in data


def test_upload_empty_document(client):
    response = client.post(
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_upload_unsupported_file_type(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "script.exe",
                b"some content",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 415


def test_upload_file_larger_than_10_mib(client):
    large_content = b"a" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/documents",
        files={
            "file": (
                "large.pdf",
                large_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413


def test_list_documents(client):
    client.post(
        "/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    response = client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["filename"] == "notes.txt"


def test_upload_blank_pdf(client):
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)

    response = client.post(
        "/documents",
        files={
            "file": (
                "blank.pdf",
                buffer.getvalue(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The uploaded PDF contains no extractable text."
    )
