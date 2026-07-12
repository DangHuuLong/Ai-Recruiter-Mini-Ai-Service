"""Heuristic section splitter for resume text."""

import logging
import re

from app.parsers.normalizer import strip_accents, strip_list_marker

logger = logging.getLogger(__name__)


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
        "about",
        "introduction",
        "tom tat",
        "gioi thieu",
        "giới thiệu",
        "muc tieu nghe nghiep",
    ),
    "skills": (
        "skills",
        "technical skills",
        "core skills",
        "personal skill",
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
        "career history",
        "career histories",
        "employment",
        "kinh nghiem",
        "kinh nghiem lam viec",
        "qua trinh lam viec",
    ),
    "projects": (
        "projects",
        "project",
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
        "hoc tap",
        "qua trinh hoc tap",
        "bang cap",
    ),
    "certifications": (
        "certification",
        "certifications",
        "certificate",
        "certificates",
        "certicate",
        "certicates",
        "licenses",
        "chung chi",
    ),
    "achievements": (
        "achievements",
        "achievement",
        "awards",
        "honors",
        "honours",
        "honors awards",
        "honours awards",
        "honors and awards",
        "honours and awards",
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

PROJECT_FIELD_LABELS = {
    "achievement/skills",
    "cong nghe su dung",
    "description",
    "key contributions",
    "key responsibilities",
    "main duties",
    "mo ta chuc nang",
    "position",
    "role",
    "team size",
    "teamsize",
    "technologies",
    "technology",
    "tech stack",
    "vai tro",
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


def matches_known_header(line: str) -> bool:
    """Whether the regex/alias splitter recognizes `line` as a section header.

    Exposed as a small public signal (rather than requiring callers to reach
    into `_match_header_line`) so the ML section-classifier's feature
    extractor (app/ml/section_classifier_features.py) can feed the existing
    heuristic's own confidence in as a feature, instead of duplicating the
    alias-matching logic.
    """
    return _match_header_line(line) is not None


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


def _is_project_field_line(line: str) -> bool:
    if ":" not in line:
        return False

    label, _ = line.split(":", 1)
    return _normalize_heading(label) in PROJECT_FIELD_LABELS


def find_headings(text: str) -> list[re.Match[str]]:
    pattern = r"^\s*(?:[-*]\s*)?(?P<h>[A-Za-z0-9&/+.# \-]{2,70})\s*:?\s*$"
    return [
        match
        for match in re.finditer(pattern, text or "", flags=re.MULTILINE)
        if _match_header_line(match.group(0))
    ]


def _regex_split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {key: [] for key in SECTION_ORDER}
    sections["other"] = []

    current = "other"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        # In a Projects section, rows such as "Technologies: ReactJS" or
        # "Key Responsibilities:" are project-field labels, not top-level resume
        # section headers. Without this guard, the splitter moves subsequent
        # project text into Skills and only the first project title survives.
        if current == "projects" and _is_project_field_line(line):
            sections[current].append(line)
            continue

        header_match = _match_header_line(line)
        if header_match:
            current, inline_content = header_match
            if inline_content:
                sections[current].append(inline_content)
            continue

        sections[current].append(line)

    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _ml_assisted_split(text: str) -> dict[str, str]:
    """Regex-first, ML-fallback-for-ambiguous-lines only.

    A header match is a high-precision signal, so any line the regex pass
    confidently assigns to a real section is kept as-is. The regex splitter
    can only ever leave a line in "other" for lines seen BEFORE the first
    recognized header (or the whole document, if no header is ever
    recognized) — headers themselves always move `current` to a real
    section, so "other" is exactly the heuristic's blind spot. This function
    reclassifies just that blind spot with the distilled section-classifier
    model (see app/ml/section_classifier_model.py), Viterbi-smoothed so the
    reclassified lines still form coherent runs rather than flip-flopping
    line by line.

    Raises whatever get_section_classifier_model() raises when the model is
    disabled or its artifacts are missing — callers (split_sections) must
    catch that and fall back to _regex_split_sections.
    """
    from app.ml.section_classifier_model import get_section_classifier_model

    sections = _regex_split_sections(text)
    other_text = sections.get("other", "")
    if not other_text:
        return sections

    other_lines = other_text.splitlines()
    bundle = get_section_classifier_model()
    predicted_labels = bundle.predict_labels(other_lines)

    remaining_other: list[str] = []
    additions: dict[str, list[str]] = {}
    for line, label in zip(other_lines, predicted_labels):
        if label == "other":
            remaining_other.append(line)
        else:
            additions.setdefault(label, []).append(line)

    merged = dict(sections)
    for label, lines in additions.items():
        # These lines occurred earliest in the document (before any
        # recognized header), so they're prepended ahead of whatever content
        # the regex pass already assigned to this section.
        prefix = "\n".join(lines)
        existing = merged.get(label, "")
        merged[label] = f"{prefix}\n{existing}".strip() if existing else prefix
    merged["other"] = "\n".join(remaining_other).strip()

    return merged


def split_sections(text: str) -> dict[str, str]:
    try:
        return _ml_assisted_split(text)
    except Exception:
        logger.debug("ML-assisted section split unavailable, falling back to regex.", exc_info=True)
        return _regex_split_sections(text)
