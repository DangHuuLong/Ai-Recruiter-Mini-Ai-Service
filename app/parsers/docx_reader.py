"""DOCX text extraction helpers."""

from pathlib import Path
from typing import Any


def _load_document_factory() -> Any:
    try:
        from docx import Document

        return Document
    except Exception as exc:  # pragma: no cover - dependency error path
        raise ImportError("python-docx is required to read DOCX files") from exc


def _iter_table_text(document: Any) -> list[str]:
    values = []
    for table in getattr(document, "tables", []):
        for row in getattr(table, "rows", []):
            for cell in getattr(row, "cells", []):
                text = getattr(cell, "text", "").strip()
                if text:
                    values.append(text)
    return values


def read_docx(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    Document = _load_document_factory()
    document = Document(str(file_path))

    parts = [paragraph.text.strip() for paragraph in getattr(document, "paragraphs", []) if paragraph.text.strip()]
    parts.extend(_iter_table_text(document))

    return "\n".join(parts)
