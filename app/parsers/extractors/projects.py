import re

from app.parsers.extractors.projects_certifications import (
    DATE_RANGE_RE,
    FEATURE_STARTERS,
    LOOSE_DATE_RANGE_RE,
    _is_valid_project,
    _looks_like_project_title,
    _strip_date_suffix,
    parse_project_block,
)
from app.parsers.normalizer import strip_accents, strip_list_marker
from app.parsers.normalizers.duration_normalizer import extract_date_range

LOOSE_DATED_PROJECT_TITLE_RE = re.compile(
    r"^(?P<title>.+?)\s+"
    r"(?P<start>(?:19|20)\d{2}(?:[/.-]\d{1,2})?|\d{1,2}[/.-](?:19|20)\d{2})"
    r"\s*(?:-|–|—|to|den|đến)\s*"
    r"(?P<end>nay|present|current|now|hien tai|hiện tại|(?:19|20)\d{2}(?:[/.-]\d{1,2})?|\d{1,2}[/.-](?:19|20)\d{2})\s*$",
    re.IGNORECASE,
)

SINGLE_DATE_RE = re.compile(
    r"^(?:(?:19|20)\d{2}(?:[/.-]\d{1,2})?|\d{1,2}[/.-](?:19|20)\d{2})$",
    re.IGNORECASE,
)

URL_ONLY_RE = re.compile(r"^(?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}(?:/[\w\-./?=#%&+]*)?/?$", re.IGNORECASE)

ROLE_LINE_TITLES = {
    "backend developer",
    "developer",
    "frontend developer",
    "full stack developer",
    "fullstack developer",
    "game developer",
    "intern fullstack developer",
    "junior web developer",
    "personal project",
    "web developer",
}

PROJECT_TITLE_KEYWORDS = (
    "app",
    "application",
    "booking",
    "cinema",
    "clone",
    "commerce",
    "construction",
    "e-commerce",
    "ecommerce",
    "learning",
    "management",
    "meeting",
    "mobile",
    "platform",
    "portfolio",
    "portal",
    "project",
    "recruiter",
    "recruitment",
    "shop",
    "site",
    "sneaker",
    "store",
    "system",
    "website",
)

PROJECT_CONTEXT_TOKENS = (
    "achievement/skills",
    "built ",
    "cong nghe su dung",
    "designed ",
    "description",
    "developed ",
    "du an ca nhan",
    "du an nhom",
    "implemented ",
    "integrated ",
    "key contributions",
    "key responsibilities",
    "link github",
    "main duties",
    "mo ta chuc nang",
    "personal project",
    "position:",
    "role:",
    "team size",
    "teamsize",
    "technologies",
    "technology",
    "tech stack",
    "vai tro",
)

IMMEDIATE_PROJECT_CONTEXT_TOKENS = (
    "built ",
    "cong nghe su dung",
    "designed ",
    "description:",
    "developed ",
    "du an ca nhan",
    "du an nhom",
    "implemented ",
    "integrated ",
    "link github",
    "personal project",
    "position:",
    "role:",
    "team size",
    "teamsize",
    "technologies",
    "technology",
    "tech stack",
)

PROJECT_DETAIL_LABELS = (
    "achievement/skills",
    "description",
    "key contributions",
    "key responsibilities",
    "main duties",
    "personal project",
    "position",
    "role",
    "team size",
    "teamsize",
    "technologies",
    "technology",
    "tech stack",
)

SECTION_NOISE_TITLES = {
    "about",
    "address",
    "birthday",
    "contact",
    "email",
    "gender",
    "hobbies",
    "phone",
    "profile",
    "references",
    "skills",
}

TERMINAL_SECTION_TITLES = {
    "academic",
    "academic background",
    "awards",
    "bang cap",
    "certicate",
    "certificate",
    "certification",
    "database",
    "education",
    "education background",
    "frontend",
    "hoc tap",
    "hoc van",
    "honors",
    "honors awards",
    "honors and awards",
    "honours awards",
    "honours and awards",
    "language",
    "languages",
    "objective",
    "personal skill",
    "soft skills",
    "summary",
    "tools",
    "work experience",
}

NON_PROJECT_TITLES = SECTION_NOISE_TITLES | TERMINAL_SECTION_TITLES

SKILL_NOISE_PATTERNS = (
    "adobe",
    "adobe photoshop",
    "adobe xd",
    "angularjs",
    "html/css",
    "javascript framework",
    "photoshop",
    "reactjs/angularjs",
    "vuejs",
)

REFERENCE_NOISE_RE = re.compile(
    r"\b(project manager|phone\s*:|email\s*:|@|one\s*tech|onetech)\b",
    re.IGNORECASE,
)

CONTACT_NOISE_RE = re.compile(
    r"^(?:\+?\d[\d\s().-]{5,}|male|female|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})$",
    re.IGNORECASE,
)


def _normalize_title(value: str) -> str:
    normalized = strip_accents(value or "").lower()
    normalized = re.sub(r"[^a-z0-9+#./ ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_feature_name(line: str) -> bool:
    normalized = _normalize_title(line)
    return any(normalized.startswith(prefix) for prefix in FEATURE_STARTERS)


def _is_url_only_line(line: str) -> bool:
    return bool(URL_ONLY_RE.fullmatch(strip_list_marker(line).strip()))


def _is_role_line(line: str) -> bool:
    return _normalize_title(_strip_date_suffix(strip_list_marker(line).strip())) in ROLE_LINE_TITLES


def _is_project_detail_line(line: str) -> bool:
    normalized = _normalize_title(line)
    return any(normalized.startswith(label) for label in PROJECT_DETAIL_LABELS) or _is_role_line(line)


def _is_skill_noise(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    normalized = _normalize_title(clean)
    if not normalized:
        return False

    if _is_project_detail_line(clean) or ":" in clean:
        return False

    if len(normalized.split()) > 4:
        return False

    return normalized in SKILL_NOISE_PATTERNS or any(pattern in normalized for pattern in SKILL_NOISE_PATTERNS)


def _is_reference_noise(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    normalized = _normalize_title(clean)
    if not clean:
        return False
    if REFERENCE_NOISE_RE.search(clean):
        return True
    return normalized.endswith(" project manager") or " project manager " in normalized


def _is_contact_noise(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    return bool(CONTACT_NOISE_RE.fullmatch(clean))


def _is_non_project_title(line: str) -> bool:
    clean = _strip_date_suffix(strip_list_marker(line).strip())
    normalized = _normalize_title(clean)
    return (
        normalized in NON_PROJECT_TITLES
        or _is_feature_name(clean)
        or _is_skill_noise(clean)
        or _is_reference_noise(clean)
        or _is_contact_noise(clean)
    )


def _is_terminal_stop_line(line: str) -> bool:
    clean = _strip_date_suffix(strip_list_marker(line).strip())
    return _normalize_title(clean) in TERMINAL_SECTION_TITLES


def _is_layout_noise_line(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    normalized = _normalize_title(clean)
    return (
        normalized in SECTION_NOISE_TITLES
        or _is_skill_noise(clean)
        or _is_reference_noise(clean)
        or _is_contact_noise(clean)
    )


def _is_date_only_line(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    return bool(
        DATE_RANGE_RE.fullmatch(clean)
        or LOOSE_DATE_RANGE_RE.fullmatch(clean)
        or SINGLE_DATE_RE.fullmatch(clean)
        or re.fullmatch(r"(?:19|20)\d{2}(?:[/.-]\d{1,2})?", clean)
    )


def _split_content_and_date_only_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    content_lines: list[str] = []
    date_lines: list[str] = []

    for line in lines:
        clean = strip_list_marker(line).strip()
        if _is_date_only_line(clean):
            date_lines.append(clean)
        else:
            content_lines.append(line)

    return content_lines, date_lines


def _coerce_loose_date_range(line: str) -> str:
    match = LOOSE_DATE_RANGE_RE.fullmatch(strip_list_marker(line).strip())
    if not match:
        return line
    return f"{match.group('start')} - {match.group('end')}"


def _apply_trailing_dates(projects: list[dict], date_lines: list[str]) -> None:
    if not date_lines or len(date_lines) < len(projects):
        return

    for project, date_line in zip(projects, date_lines, strict=False):
        if project.get("start_date") or project.get("end_date"):
            continue

        date_range = extract_date_range(_coerce_loose_date_range(date_line))
        if not date_range:
            continue

        project["start_date"] = date_range.get("start_date")
        project["end_date"] = date_range.get("end_date")


def _has_project_keyword(value: str) -> bool:
    lowered = strip_accents(value or "").lower()
    return any(keyword in lowered for keyword in PROJECT_TITLE_KEYWORDS)


def _has_project_context(next_lines: list[str]) -> bool:
    window = "\n".join(strip_accents(strip_list_marker(line)).lower() for line in next_lines[:8])
    return any(token in window for token in PROJECT_CONTEXT_TOKENS) or bool(DATE_RANGE_RE.search(window) or LOOSE_DATE_RANGE_RE.search(window))


def _has_strong_stacked_project_context(next_lines: list[str]) -> bool:
    meaningful_lines = [line for line in next_lines[:6] if not _is_layout_noise_line(line)]
    if not meaningful_lines:
        return False

    first = strip_accents(strip_list_marker(meaningful_lines[0])).lower()
    if _is_url_only_line(meaningful_lines[0]) or _is_date_only_line(meaningful_lines[0]):
        return True

    if any(token in first for token in IMMEDIATE_PROJECT_CONTEXT_TOKENS):
        return True

    if _is_role_line(meaningful_lines[0]):
        return _has_project_context(meaningful_lines[1:])

    if len(meaningful_lines) >= 2 and _is_url_only_line(meaningful_lines[0]) and _is_role_line(meaningful_lines[1]):
        return _has_project_context(meaningful_lines[2:])

    return False


def _is_project_context_line(line: str) -> bool:
    normalized = _normalize_title(line)
    return any(normalized.startswith(token.rstrip(":")) for token in PROJECT_CONTEXT_TOKENS)


def _looks_like_inline_project_header(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if _is_non_project_title(clean) or _is_project_context_line(clean):
        return False

    for pattern in (r"\s+[-–—]\s+", r":\s+"):
        parts = re.split(pattern, clean, maxsplit=1)
        if len(parts) != 2:
            continue

        name = parts[0].strip(" -–—|,")
        rest = parts[1].strip()
        full_title = f"{name} {rest}".strip()
        if not name or _is_non_project_title(name) or _is_project_context_line(name):
            continue

        if not (_looks_like_project_title(name) or _looks_like_project_title(full_title)):
            continue

        normalized_rest = strip_accents(rest).lower()
        if _has_project_keyword(full_title) or any(token in normalized_rest for token in PROJECT_CONTEXT_TOKENS):
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


def _looks_like_stacked_project_title(line: str, next_lines: list[str]) -> bool:
    clean = strip_list_marker(line).strip()
    if _is_non_project_title(clean) or _is_date_only_line(clean) or _is_project_context_line(clean) or _is_role_line(clean):
        return False

    if "," in clean:
        return False

    if not _looks_like_project_title(clean):
        return False

    if _has_project_keyword(clean):
        return _has_strong_stacked_project_context(next_lines) or _has_project_context(next_lines)

    return _has_strong_stacked_project_context(next_lines)


def _should_start_new_block(line: str, current: list[str], next_lines: list[str]) -> bool:
    clean = strip_list_marker(line).strip()

    if _is_non_project_title(clean) or _is_date_only_line(clean) or _is_project_context_line(clean) or _is_role_line(clean):
        return False

    if _is_dated_project_title(clean) or _looks_like_inline_project_header(clean):
        return True

    if _looks_like_stacked_project_title(clean, next_lines):
        return True

    return False


def _clean_project_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    skipping_layout_noise = False

    for line in lines:
        if _is_terminal_stop_line(line):
            break

        if _is_layout_noise_line(line):
            skipping_layout_noise = True
            continue

        if skipping_layout_noise and not (_is_project_detail_line(line) or _is_dated_project_title(line) or _is_url_only_line(line)):
            continue

        skipping_layout_noise = False
        cleaned.append(line)

    return cleaned


def split_project_blocks(text: str) -> list[list[str]]:
    lines = [strip_list_marker(raw) for raw in (text or "").splitlines()]
    lines = [line for line in lines if line]

    blocks: list[list[str]] = []
    current: list[str] = []

    for index, line in enumerate(lines):
        next_lines = lines[index + 1 : index + 9]

        if current and _is_terminal_stop_line(line):
            cleaned = _clean_project_lines(current)
            if cleaned:
                blocks.append(cleaned)
            current = []
            continue

        if current and _is_layout_noise_line(line):
            current.append(line)
            continue

        if current and (
            _is_dated_project_title(line)
            or _looks_like_inline_project_header(line)
            or _looks_like_stacked_project_title(line, next_lines)
        ):
            cleaned = _clean_project_lines(current)
            if cleaned:
                blocks.append(cleaned)
            current = [line]
            continue

        if _should_start_new_block(line, current, next_lines):
            if current:
                cleaned = _clean_project_lines(current)
                if cleaned:
                    blocks.append(cleaned)

            current = [line]
            continue

        if current:
            current.append(line)

    if current:
        cleaned = _clean_project_lines(current)
        if cleaned:
            blocks.append(cleaned)

    return blocks


def _extract_dated_project_blocks(text: str) -> list[list[str]]:
    lines = [strip_list_marker(raw) for raw in (text or "").splitlines()]
    lines = [line for line in lines if line]
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if _is_dated_project_title(line):
            if current:
                cleaned = _clean_project_lines(current)
                if cleaned:
                    blocks.append(cleaned)
            current = [line]
            continue

        if current:
            if _is_terminal_stop_line(line):
                cleaned = _clean_project_lines(current)
                if cleaned:
                    blocks.append(cleaned)
                current = []
                continue
            current.append(line)

    if current:
        cleaned = _clean_project_lines(current)
        if cleaned:
            blocks.append(cleaned)

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


def _is_valid_project_item(project: dict) -> bool:
    if not _is_valid_project(project):
        return False

    name = project.get("name") or ""
    return not _is_non_project_title(name) and not _is_date_only_line(name)


def extract_projects(text: str) -> list[dict]:
    raw_lines = [strip_list_marker(raw) for raw in (text or "").splitlines()]
    raw_lines = [line for line in raw_lines if line]
    content_lines, date_only_lines = _split_content_and_date_only_lines(raw_lines)
    content_text = "\n".join(content_lines)
    projects = []

    for block in split_project_blocks(content_text):
        project = parse_project_block(block)
        if _is_valid_project_item(project):
            projects.append(project)

    for block in _extract_dated_project_blocks(text):
        project = parse_project_block(block)
        if _is_valid_project_item(project):
            projects.append(project)

    projects = _dedupe_projects(projects)
    _apply_trailing_dates(projects, date_only_lines)
    return projects
