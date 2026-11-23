"""PDF -> pages -> overlapping chunks that remember their page number (for citations)."""
from __future__ import annotations

import io
import re
import uuid

from pypdf import PdfReader

from app.vectorstore import Chunk


class IngestError(ValueError):
    pass


def extract_pages(pdf_bytes: bytes) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as e:
        raise IngestError(f"Could not read PDF: {e}") from e
    if not any(p.strip() for p in pages):
        raise IngestError("No text found. Scanned/image-only PDFs need OCR first.")
    return pages


def clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on word boundaries into ~`size`-character windows overlapping by ~`overlap`."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        if length + len(word) + 1 > size and current:
            chunks.append(" ".join(current))
            # carry the tail of the previous chunk forward as overlap
            tail, tail_len = [], 0
            for w in reversed(current):
                if tail_len + len(w) + 1 > overlap:
                    break
                tail.insert(0, w)
                tail_len += len(w) + 1
            current, length = tail, tail_len
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[str], document_id: int, filename: str, size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for page_no, raw in enumerate(pages, start=1):
        text = clean(raw)
        if not text:
            continue
        for piece in split_text(text, size, overlap):
            chunks.append(
                Chunk(id=uuid.uuid4().hex, text=piece, document_id=document_id, filename=filename, page=page_no)
            )
    return chunks
