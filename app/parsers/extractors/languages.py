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
    "good",
)

LANGUAGE_SKILL_KEYWORDS = (
    "listening",
    "speaking",
    "reading",
    "writing",
    "communicate",
    "communication",
    "remote working",
    "working environments",
)

TOEIC_RE = re.compile(r"\bTOEIC\s*[:\-]?\s*(?P<score>\d{3,4})\s*(?:/\s*(?P<scale>\d{3,4}))?\b", re.IGNORECASE)


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


def _format_toeic(match: re.Match[str]) -> str:
    score = match.group("score")
    scale = match.group("scale")
    return f"TOEIC {score}/{scale}" if scale else f"TOEIC {score}"


def _extract_toeic_proficiency(value: str) -> str | None:
    match = TOEIC_RE.search(value or "")
    return _format_toeic(match) if match else None


def _extract_proficiency(value: str) -> str | None:
    toeic = _extract_toeic_proficiency(value)
    if toeic:
        return toeic

    normalized = _normalize_language_key(value)
    for word in PROFICIENCY_WORDS:
        if re.search(rf"\b{word}\b", normalized):
            return word

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
        line = line.lstrip('"').strip()
        if not line:
            continue

        normalized = _normalize_language_key(line)
        starts_language = _extract_language_name(line) is not None
        looks_language_continuation = any(keyword in normalized for keyword in LANGUAGE_SKILL_KEYWORDS)

        if merged_lines and not starts_language and looks_language_continuation:
            merged_lines[-1] = f"{merged_lines[-1]} {line}".strip()
        else:
            merged_lines.append(line)

    return merged_lines


def _recover_vietnamese_english_proficiency(text: str) -> str | None:
    lines = [strip_list_marker(raw_line).lstrip('"').strip() for raw_line in (text or "").splitlines()]
    lines = [line for line in lines if line]

    for index, line in enumerate(lines):
        normalized = _normalize_language_key(line)
        if not normalized.startswith("tieng anh"):
            continue

        candidates = [line]
        if index + 1 < len(lines):
            next_line = lines[index + 1]
            normalized_next = _normalize_language_key(next_line)
            if normalized_next.startswith(("va ", "and ")) or not _extract_language_name(next_line):
                candidates.append(next_line)

        combined = " ".join(candidates)
        if ":" in combined:
            _, proficiency = combined.split(":", 1)
            return re.sub(r"\s+", " ", proficiency.strip(" -|,")) or None

    return None


def _upsert_language(languages: list[dict], seen: set[str], name: str, proficiency: str | None) -> None:
    if name in seen:
        if proficiency:
            for language in languages:
                if language.get("name") == name and not language.get("proficiency"):
                    language["proficiency"] = proficiency
                    break
        return

    seen.add(name)
    languages.append({"name": name, "proficiency": proficiency})


def extract_languages(text: str) -> list[dict]:
    languages = []
    seen = set()

    for line in _merge_wrapped_language_lines(text):
        parts = [part.strip() for part in re.split(r"[,;]", line) if part.strip()]
        if ":" in line:
            parts = [line]

        for part in parts:
            name = _extract_language_name(part)
            if not name:
                continue

            proficiency = _extract_proficiency(part)
            if name == "English":
                proficiency = _recover_vietnamese_english_proficiency(text) or proficiency

            _upsert_language(languages, seen, name, proficiency)

    toeic = _extract_toeic_proficiency(text or "")
    if toeic:
        _upsert_language(languages, seen, "English", toeic)

    return languages
