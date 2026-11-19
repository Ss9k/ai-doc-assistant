import re
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
    """Split text into overlapping chunks using natural text boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Search for natural boundaries near the chunk end.
            search_start = max(start + 1, end - 150)

            boundaries = [
                (text.rfind("\n\n", search_start, end), 2),
                (text.rfind("\n", search_start, end), 1),
                (text.rfind(".", search_start, end), 1),
                (text.rfind(" ", search_start, end), 1),
            ]

            valid_boundaries = [
                position + length
                for position, length in boundaries
                if position >= search_start
            ]

            if valid_boundaries:
                end = max(valid_boundaries)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Preserve the requested overlap.
        next_start = max(start + 1, end - overlap)

        # Align to the next word boundary when possible.
        match = re.search(r"\s+", text[next_start:])

        if match:
            next_start += match.end()

        # If there is no whitespace, retain the calculated position.
        start = next_start

    return chunks
