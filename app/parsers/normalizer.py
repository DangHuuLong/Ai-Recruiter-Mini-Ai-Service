"""Text normalization utilities for resume parsing."""

import re
import unicodedata


def strip_accents(text: str) -> str:
    value = text or ""
    value = value.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t\f\v]+", " ", text or "").strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_accents: bool = True,
    preserve_lines: bool = False,
) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if remove_accents:
        normalized = strip_accents(normalized)
    if lowercase:
        normalized = normalized.lower()

    if preserve_lines:
        lines = [normalize_spaces(line) for line in normalized.split("\n")]
        return "\n".join(line for line in lines if line)

    return normalize_whitespace(normalized)


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def strip_list_marker(line: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\u2022|\d+[.)])\s+", "", line or "").strip()
