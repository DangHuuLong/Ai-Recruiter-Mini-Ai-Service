from app.parsers.extractors.projects_certifications import (
    DATE_RANGE_RE,
    _has_strong_project_header_context,
    _is_valid_project,
    _looks_like_project_title,
    _strip_date_suffix,
    parse_project_block,
)
from app.parsers.normalizer import strip_list_marker


def _is_dated_project_title(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if not DATE_RANGE_RE.search(clean):
        return False

    title = _strip_date_suffix(clean)
    return title != clean and _looks_like_project_title(title)


def _should_start_new_block(line: str, current: list[str], next_lines: list[str]) -> bool:
    clean = strip_list_marker(line).strip()

    if not _looks_like_project_title(clean):
        return False

    if not current:
        return True

    if _is_dated_project_title(clean):
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


def extract_projects(text: str) -> list[dict]:
    projects = []

    for block in split_project_blocks(text):
        project = parse_project_block(block)
        if _is_valid_project(project):
            projects.append(project)

    return projects
