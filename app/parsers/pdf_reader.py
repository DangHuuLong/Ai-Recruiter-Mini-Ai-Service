"""PDF text extraction helpers."""

from pathlib import Path
from typing import Any


def _load_pdf_reader() -> Any:
    try:
        from pypdf import PdfReader

        return PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader

            return PdfReader
        except Exception as exc:  # pragma: no cover - dependency error path
            raise ImportError("pypdf or PyPDF2 is required to read PDF files") from exc


def read_pdf(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    PdfReader = _load_pdf_reader()
    reader = PdfReader(str(file_path))

    text_parts = []
    for page in getattr(reader, "pages", []):
        text_parts.append(page.extract_text() or "")

    return "\n".join(part.strip() for part in text_parts if part.strip())
