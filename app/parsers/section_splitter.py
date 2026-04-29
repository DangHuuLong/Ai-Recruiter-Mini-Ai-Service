"""Heuristic section splitter for resume text."""

import re

from app.parsers.normalizer import strip_accents, strip_list_marker


SECTION_ORDER = (
    "summary",
    "skills",
    "experience",
    "projects",
    "education",
    "certifications",
    "achievements",
    "languages",
)

HEADERS = {
    "summary": (
        "summary",
        "profile",
        "professional summary",
        "career objective",
        "objective",
        "tom tat",
        "muc tieu nghe nghiep",
    ),
    "skills": (
        "skills",
        "technical skills",
        "core skills",
        "technologies",
        "tech stack",
        "ky nang",
        "ky nang chuyen mon",
    ),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "working experience",
        "employment history",
        "kinh nghiem",
        "kinh nghiem lam viec",
        "qua trinh lam viec",
    ),
    "projects": (
        "projects",
        "personal projects",
        "selected projects",
        "project experience",
        "du an",
        "du an ca nhan",
    ),
    "education": (
        "education",
        "academic",
        "academic background",
        "education background",
        "hoc van",
        "bang cap",
    ),
    "certifications": (
        "certification",
        "certifications",
        "certificates",
        "licenses",
        "chung chi",
    ),
    "achievements": (
        "achievements",
        "awards",
        "honors",
        "awards and achievements",
        "thanh tich",
        "giai thuong",
    ),
    "languages": (
        "languages",
        "language",
        "ngon ngu",
    ),
}


def _normalize_heading(value: str) -> str:
    normalized = strip_accents(value).lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9+#./ ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


HEADER_LOOKUP = {
    _normalize_heading(alias): section
    for section, aliases in HEADERS.items()
    for alias in aliases
}


def _match_header_line(line: str) -> tuple[str, str] | None:
    candidate = strip_list_marker(line).strip("# ").strip()
    if not candidate:
        return None

    header_text = candidate
    inline_content = ""
    if ":" in candidate:
        header_text, inline_content = candidate.split(":", 1)
        header_text = header_text.strip()
        inline_content = inline_content.strip()
    elif len(candidate.split()) > 5:
        return None

    section = HEADER_LOOKUP.get(_normalize_heading(header_text))
    if not section:
        return None

    return section, inline_content


def find_headings(text: str) -> list[re.Match[str]]:
    pattern = r"^\s*(?:[-*]\s*)?(?P<h>[A-Za-z0-9&/+.# \-]{2,70})\s*:?\s*$"
    return [
        match
        for match in re.finditer(pattern, text or "", flags=re.MULTILINE)
        if _match_header_line(match.group(0))
    ]


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {key: [] for key in SECTION_ORDER}
    sections["other"] = []

    current = "other"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        header_match = _match_header_line(line)
        if header_match:
            current, inline_content = header_match
            if inline_content:
                sections[current].append(inline_content)
            continue

        sections[current].append(line)

    return {key: "\n".join(value).strip() for key, value in sections.items()}
