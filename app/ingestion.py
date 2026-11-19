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
    """Split text into overlapping chunks without breaking normal words."""
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

        if max_end == len(text):
            end = max_end
        else:
            # Prefer the last whitespace boundary within chunk_size.
            boundaries = list(re.finditer(r"\s+", text[start:max_end]))

            if boundaries:
                end = start + boundaries[-1].start()
            else:
                # Handle a token longer than chunk_size.
                end = max_end

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        # Calculate the next starting position with overlap.
        candidate = max(start + 1, end - overlap)

        # Align to a word boundary when possible.
        previous_boundaries = list(re.finditer(r"\s+", text[start:candidate]))

        if previous_boundaries:
            next_start = start + previous_boundaries[-1].end()
        else:
            # No whitespace: continue splitting an oversized token.
            next_start = candidate

        start = next_start

    return chunks
