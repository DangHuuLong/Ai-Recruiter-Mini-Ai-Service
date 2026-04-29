"""Simple DOCX reader wrapper.
Uses python-docx if available.
"""

def read_docx(path: str) -> str:
    try:
        import docx
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("python-docx is required to read DOCX files") from exc

    doc = docx.Document(path)
    parts: list[str] = []
    for p in doc.paragraphs:
        parts.append(p.text)

    return "\n".join(parts)
