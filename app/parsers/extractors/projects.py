import re

from app.parsers.extractors.projects_certifications import (
    DATE_RANGE_RE,
    _has_strong_project_header_context,
    _is_valid_project,
    _looks_like_project_title,
    _strip_date_suffix,
    parse_project_block,
)
from app.parsers.normalizer import strip_accents, strip_list_marker

LOOSE_DATED_PROJECT_TITLE_RE = re.compile(
    r"^(?P<title>.+?)\s+"
    r"(?P<start>(?:19|20)\d{2}(?:[/.-]\d{1,2})?|\d{1,2}[/.-](?:19|20)\d{2})"
    r"\s*(?:-|–|—|to|den|đến)\s*"
    r"(?P<end>nay|present|current|now|hien tai|hiện tại|(?:19|20)\d{2}(?:[/.-]\d{1,2})?|\d{1,2}[/.-](?:19|20)\d{2})\s*$",
    re.IGNORECASE,
)

PROJECT_TITLE_KEYWORDS = (
    "app",
    "application",
    "e-commerce",
    "ecommerce",
    "platform",
    "portal",
    "project",
    "site",
    "system",
    "website",
)

NON_PROJECT_TITLES = {
    "academic",
    "academic background",
    "bang cap",
    "certicate",
    "certificate",
    "certification",
    "contact",
    "education",
    "education background",
    "hoc tap",
    "hoc van",
    "language",
    "languages",
    "objective",
    "profile",
    "summary",
}


def _normalize_title(value: str) -> str:
    normalized = strip_accents(value or "").lower()
    normalized = re.sub(r"[^a-z0-9+#./ ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_non_project_title(line: str) -> bool:
    clean = _strip_date_suffix(strip_list_marker(line).strip())
    return _normalize_title(clean) in NON_PROJECT_TITLES


def _looks_like_loose_dated_project_title(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if _is_non_project_title(clean):
        return False

    match = LOOSE_DATED_PROJECT_TITLE_RE.match(clean)
    if not match:
        return False

    title = match.group("title").strip(" -|,")
    lowered_title = title.lower()

    if _is_non_project_title(title):
        return False

    if not _looks_like_project_title(title):
        return False

    return any(keyword in lowered_title for keyword in PROJECT_TITLE_KEYWORDS)


def _is_dated_project_title(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if _is_non_project_title(clean):
        return False

    if _looks_like_loose_dated_project_title(clean):
        return True

    if not DATE_RANGE_RE.search(clean):
        return False

    title = _strip_date_suffix(clean)
    return title != clean and not _is_non_project_title(title) and _looks_like_project_title(title)


def _should_start_new_block(line: str, current: list[str], next_lines: list[str]) -> bool:
    clean = strip_list_marker(line).strip()

    if _is_non_project_title(clean):
        return False

    if _is_dated_project_title(clean):
        return True

    if not _looks_like_project_title(clean):
        return False

    if not current:
        return True

    if next_lines and _has_strong_project_header_context(next_lines[0]):
        return True

    if len(next_lines) >= 2 and len(next_lines[0].split()) <= 4:
        return _has_strong_project_header_context(next_lines[1])

    return False


def split_project_blocks(text: str) -> list[list[str]]:
    lines = [strip_list_marker(raw) for raw in (text or "").splitlines()]
    lines = [line for line in lines if line]

    blocks: list[list[str]] = []
    current: list[str] = []

    for index, line in enumerate(lines):
        next_lines = lines[index + 1 : index + 4]

        if current and _is_dated_project_title(line):
            blocks.append(current)
            current = [line]
            continue

        if _should_start_new_block(line, current, next_lines):
            if current:
                blocks.append(current)

            current = [line]
            continue

        if current:
            current.append(line)
        else:
            current = [line]

    if current:
        blocks.append(current)

    return blocks


def _dedupe_projects(projects: list[dict]) -> list[dict]:
    unique_projects = []
    seen_names = set()

    for project in projects:
        name = project.get("name")
        normalized_name = name.lower().strip() if isinstance(name, str) else ""
        if not normalized_name or normalized_name in seen_names:
            continue

        seen_names.add(normalized_name)
        unique_projects.append(project)

    return unique_projects


def extract_projects(text: str) -> list[dict]:
    projects = []

    for block in split_project_blocks(text):
        project = parse_project_block(block)
        if _is_valid_project(project):
            projects.append(project)

    return _dedupe_projects(projects)
