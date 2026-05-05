import re

from app.parsers.normalizer import strip_accents, strip_list_marker


LANGUAGE_NAMES = {
    "english": "English",
    "vietnamese": "Vietnamese",
    "tieng anh": "English",
    "tieng viet": "Vietnamese",
    "japanese": "Japanese",
    "korean": "Korean",
    "chinese": "Chinese",
    "french": "French",
    "german": "German",
}

PROFICIENCY_WORDS = (
    "native",
    "fluent",
    "advanced",
    "intermediate",
    "basic",
    "beginner",
    "professional",
    "conversational",
)


def _normalize_language_key(value: str) -> str:
    text = strip_accents(value).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_language_name(value: str) -> str | None:
    normalized = _normalize_language_key(value)
    for key, name in LANGUAGE_NAMES.items():
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            return name
    return None


def _extract_proficiency(value: str) -> str | None:
    normalized = _normalize_language_key(value)
    for word in PROFICIENCY_WORDS:
        if re.search(rf"\b{word}\b", normalized):
            return word

    toeic_match = re.search(r"\bTOEIC\s*[:\-]?\s*\d{3,4}\b", value, re.IGNORECASE)
    if toeic_match:
        return re.sub(r"\s+", " ", toeic_match.group(0)).strip()

    if ":" in value:
        _, proficiency = value.split(":", 1)
        proficiency = proficiency.strip(" -|,")
        return proficiency or None

    normalized_language = _normalize_language_key(value)
    for key in LANGUAGE_NAMES:
        normalized_language = re.sub(rf"\b{re.escape(key)}\b", "", normalized_language).strip()

    return re.sub(r"\s+", " ", normalized_language).strip() or None


def _merge_wrapped_language_lines(text: str) -> list[str]:
    merged_lines: list[str] = []

    for raw_line in (text or "").splitlines():
        line = strip_list_marker(raw_line)
        if not line:
            continue

        if merged_lines and not _extract_language_name(line) and not re.search(r"[,;:]", line):
            merged_lines[-1] = f"{merged_lines[-1]} {line}".strip()
        else:
            merged_lines.append(line)

    return merged_lines


def extract_languages(text: str) -> list[dict]:
    languages = []
    seen = set()

    for line in _merge_wrapped_language_lines(text):
        parts = [part.strip() for part in re.split(r"[,;]", line) if part.strip()]
        for part in parts:
            name = _extract_language_name(part)
            if not name or name in seen:
                continue

            seen.add(name)
            languages.append(
                {
                    "name": name,
                    "proficiency": _extract_proficiency(part),
                }
            )

    return languages
