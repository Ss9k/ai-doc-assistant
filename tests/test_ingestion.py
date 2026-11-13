
import pymupdf

from app.ingestion import chunk_text, extract_pdf_pages


def test_chunk_text_returns_overlapping_chunks():
    text = "A" * 1800

    chunks = chunk_text(text, chunk_size=800, overlap=100)

    assert len(chunks) == 3
    assert len(chunks[0]) == 800
    assert len(chunks[1]) == 800
    assert len(chunks[2]) == 400
    assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_text_handles_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=100, overlap=100)


def test_extract_pdf_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page1 = pdf.new_page()
        page1.insert_text((50, 50), "This is page one.")

        page2 = pdf.new_page()
        page2.insert_text((50, 50), "This is page two.")

        pdf.save(pdf_path)

    pages = extract_pdf_pages(str(pdf_path))

    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "This is page one." in pages[0][1]
    assert pages[1][0] == 2
    assert "This is page two." in pages[1][1]
