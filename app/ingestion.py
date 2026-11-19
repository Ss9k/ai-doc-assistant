from pathlib import Path

import pymupdf
from docx import Document as WordDocument


def extract_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract non-empty text from each PDF page."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_number, text))

    return pages


def extract_document_pages(file_path: str) -> list[tuple[int, str]]:
    """Extract text from supported PDF, TXT, and DOCX files."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf_pages(str(path))

    if extension == ".txt":
        text = path.read_text(encoding="utf-8-sig").strip()
        return [(1, text)] if text else []

    if extension == ".docx":
        document = WordDocument(str(path))
        text = "\n".join(
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )
        return [(1, text)] if text else []

    raise ValueError(f"Unsupported document type: {extension}")


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks at whitespace boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        max_end = min(start + chunk_size, len(text))
        end = max_end

        if max_end < len(text):
            boundary = text.rfind(" ", start, max_end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)

        # Only align to a word boundary if a whitespace
        # boundary was found in the current chunk.
        if end < max_end or text.find(" ", start, max_end) != -1:
            while next_start < len(text) and not text[next_start - 1].isspace():
                next_start += 1

            while next_start < len(text) and text[next_start].isspace():
                next_start += 1

        start = next_start

    return chunks
