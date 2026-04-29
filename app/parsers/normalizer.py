"""Text normalization utilities for resume parsing."""
import re
import unicodedata
from typing import List


def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, *, lowercase: bool = True, remove_accents: bool = True) -> str:
    t = text
    if remove_accents:
        t = strip_accents(t)
    if lowercase:
        t = t.lower()
    t = normalize_whitespace(t)
    return t


def split_lines(text: str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]
