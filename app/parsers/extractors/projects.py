import re

from app.parsers.extractors.projects_certifications import (
    DATE_RANGE_RE,
    _extract_url_match,
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
    "management",
    "platform",
    "portfolio",
    "portal",
    "project",
    "site",
    "system",
    "website",
)

PROJECT_CONTEXT_TOKENS = (
    "cong nghe su dung",
    "description",
    "du an ca nhan",
    "du an nhom",
    "key contributions",
    "key responsibilities",
    "link github",
    "mo ta chuc nang",
    "position:",
    "role:",
    "team size",
    "teamsize",
    "technologies",
    "technology",
    "tech stack",
    "vai tro",
)

ROLE_ONLY_TITLES = {
    "backend developer",
    "developer",
    "frontend developer",
    "full stack developer",
    "fullstack developer",
    "game developer",
    "intern fullstack developer",
    "junior web developer",
    "web developer",
}

NON_PROJECT_TITLES = {
    "academic",
    "academic background",
    "about",
    "address",
    "bang cap",
    "birthday",
    "certicate",
    "certificate",
    "certification",
    "contact",
    "database",
    "education",
    "education background",
    "email",
    "frontend",
    "gender",
    "hobbies",
    "hoc tap",
    "hoc van",
    "language",
    "languages",
    "objective",
    "personal skill",
    "phone",
    "profile",
    "references",
    "skills",
    "soft skills",
    "summary",
    "tools",
    "work experience",
}

STOP_TITLES = NON_PROJECT_TITLES | {
    "awards",
    "honors",
    "honours awards",
    "key achievements",
}


def _normalize_title(value: str) -> str:
    normalized = strip_accents(value or "").lower()
    normalized = re.sub(r"[^a-z0-9+#./ ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_non_project_title(line: str) -> bool:
    clean = _strip_date_suffix(strip_list_marker(line).strip())
    normalized = _normalize_title(clean)
    return normalized in NON_PROJECT_TITLES or normalized in ROLE_ONLY_TITLES


def _is_stop_line(line: str) -> bool:
    clean = _strip_date_suffix(strip_list_marker(line).strip())
    return _normalize_title(clean) in STOP_TITLES


def _is_date_only_line(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    return bool(DATE_RANGE_RE.fullmatch(clean) or re.fullmatch(r"(?:19|20)\d{2}(?:[/.-]\d{1,2})?", clean))


def _has_project_keyword(value: str) -> bool:
    lowered = strip_accents(value or "").lower()
    return any(keyword in lowered for keyword in PROJECT_TITLE_KEYWORDS)


def _has_project_context(next_lines: list[str]) -> bool:
    window = "\n".join(strip_accents(strip_list_marker(line)).lower() for line in next_lines[:8])
    return any(token in window for token in PROJECT_CONTEXT_TOKENS)


def _is_project_context_line(line: str) -> bool:
    normalized = _normalize_title(line)
    return any(normalized.startswith(token.rstrip(":")) for token in PROJECT_CONTEXT_TOKENS)


def _looks_like_inline_project_header(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if _is_non_project_title(clean) or _is_project_context_line(clean):
        return False

    for pattern in (r"\s+-\s+", r":\s+"):
        parts = re.split(pattern, clean, maxsplit=1)
        if len(parts) != 2:
            continue

        name = parts[0].strip(" -|,")
        rest = parts[1].strip()
        if not name or _is_non_project_title(name) or _is_project_context_line(name):
            continue

        if not _looks_like_project_title(name):
            continue

        if _extract_url_match(rest) or _has_project_keyword(name) or any(token in strip_accents(rest).lower() for token in PROJECT_CONTEXT_TOKENS):
            return True

    return False


def _looks_like_loose_dated_project_title(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if _is_non_project_title(clean):
        return False

    match = LOOSE_DATED_PROJECT_TITLE_RE.match(clean)
    if not match:
        return False

    title = match.group("title").strip(" -|,")

    if _is_non_project_title(title):
        return False

    if not _looks_like_project_title(title):
        return False

    return _has_project_keyword(title)


def _is_dated_project_title(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if _is_non_project_title(clean):
        return False

    if _looks_like_loose_dated_project_title(clean):
        return True

    if not DATE_RANGE_RE.search(clean):
        return False

    title = _strip_date_suffix(clean)
    return title != clean and not _is_non_project_title(title) and _looks_like_project_title(title) and _has_project_keyword(title)


def _should_start_new_block(line: str, current: list[str], next_lines: list[str]) -> bool:
    clean = strip_list_marker(line).strip()

    if _is_non_project_title(clean) or _is_date_only_line(clean) or _is_project_context_line(clean):
        return False

    if _is_dated_project_title(clean) or _looks_like_inline_project_header(clean):
        return True

    if not _looks_like_project_title(clean):
        return False

    if not _has_project_context(next_lines):
        return False

    if not current:
        return True

    return True


def split_project_blocks(text: str) -> list[list[str]]:
    lines = [strip_list_marker(raw) for raw in (text or "").splitlines()]
    lines = [line for line in lines if line]

    blocks: list[list[str]] = []
    current: list[str] = []

    for index, line in enumerate(lines):
        next_lines = lines[index + 1 : index + 9]

        if current and _is_stop_line(line):
            blocks.append(current)
            current = []
            continue

        if current and (_is_dated_project_title(line) or _looks_like_inline_project_header(line)):
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
