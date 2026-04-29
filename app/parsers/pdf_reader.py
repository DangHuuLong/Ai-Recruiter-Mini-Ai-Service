"""Simple PDF reader wrapper.
Attempts to use PyPDF2 if available, otherwise raises ImportError.
"""
from typing import Optional


def read_pdf(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyPDF2 is required to read PDF files") from exc

    text_parts: list[str] = []
    reader = PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)

    return "\n".join(text_parts)
