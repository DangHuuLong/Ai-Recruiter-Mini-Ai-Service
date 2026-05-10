import re

from app.parsers.extractors.skills import extract_skills
from app.parsers.normalizer import strip_accents, strip_list_marker
from app.parsers.normalizers.duration_normalizer import extract_date_range, parse_explicit_duration_months


ROLE_KEYWORDS = (
    "developer",
    "engineer",
    "manager",
    "lead",
    "intern",
    "designer",
    "analyst",
    "scientist",
    "consultant",
)

COMPANY_KEYWORDS = (
    "company",
    "corporation",
    "corp",
    "ltd",
    "limited",
    "inc",
    "labs",
    "studio",
    "agency",
    "startup",
    "utilities",
)

STOP_KEYWORDS = (
    "education",
    "language",
    "languages",
    "certificate",
    "certification",
    "certicate",
    "hobbies",
    "references",
    "personal skill",
    "honours",
    "honors",
    "awards",
    "projects",
    "skills",
    "soft skills",
)

PROJECT_META_PREFIXES = (
    "position:",
    "role:",
    "teamsize:",
    "team size:",
    "description:",
    "technologies:",
    "technology:",
    "key contributions:",
    "key responsibilities:",
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


def _normalize(value: str) -> str:
    normalized = strip_accents(value or "").lower()
    normalized = re.sub(r"[^a-z0-9+#./ ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _looks_like_role(value: str) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in ROLE_KEYWORDS)


def _looks_like_company(value: str) -> bool:
    lowered = value.lower()
    if _looks_like_role(value):
        return False
    if extract_date_range(value):
        return False
    if any(keyword in lowered for keyword in COMPANY_KEYWORDS):
        return True

    words = value.split()
    if not 1 <= len(words) <= 6:
        return False
    if any(char.isdigit() for char in value):
        return False
    if any(token in lowered for token in ["experience", "project", "education", "summary", "skill"]):
        return False

    return True


def _is_stop_line(value: str) -> bool:
    return _normalize(value) in STOP_KEYWORDS


def _is_achievement_heading(value: str) -> bool:
    normalized = _normalize(value)
    return normalized in {"key achievements", "achievements", "achievement"} or normalized.startswith(
        "key achievement"
    )


def _has_project_keyword(value: str) -> bool:
    lowered = strip_accents(value or "").lower()
    return any(keyword in lowered for keyword in PROJECT_TITLE_KEYWORDS)


def _looks_like_project_block_start(line: str, next_lines: list[str]) -> bool:
    if not _has_project_keyword(line):
        return False

    if extract_date_range(line):
        return True

    return any(_normalize(next_line).startswith(PROJECT_META_PREFIXES) for next_line in next_lines[:5])


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _merge_wrapped_responsibilities(items: list[str]) -> list[str]:
    merged: list[str] = []

    for item in items:
        clean = strip_list_marker(item).strip()
        if not clean:
            continue

        if not merged:
            merged.append(clean)
            continue

        previous = merged[-1]
        should_merge = (
            previous.endswith((",", "/"))
            or previous.lower().endswith((" and", " or"))
            or clean[0].islower()
            or len(clean.split()) <= 3
        )

        if should_merge:
            merged[-1] = f"{previous} {clean}".strip()
        else:
            merged.append(clean)

    return merged


def _parse_role_company(text: str) -> tuple[str | None, str | None]:
    cleaned = re.sub(r"\s+", " ", text).strip(" -|,")
    if not cleaned:
        return None, None

    at_match = re.match(r"(?P<role>.+?)\s+at\s+(?P<company>.+)", cleaned, flags=re.IGNORECASE)
    if at_match:
        return at_match.group("role").strip(), at_match.group("company").strip(" -|,")

    parts = [part.strip(" -|,") for part in re.split(r"\s+\|\s+|\s+-\s+|,", cleaned) if part.strip(" -|,")]
    if len(parts) >= 2:
        first, second = parts[0], parts[1]
        if _looks_like_role(first):
            return first, second
        if _looks_like_role(second):
            return second, first
        return first, second

    return cleaned, None


def _parse_experience_line(line: str) -> dict:
    date_range = extract_date_range(line)
    header_text = line
    if date_range:
        header_text = header_text.replace(str(date_range["raw"]), " ")

    role, company = _parse_role_company(header_text)
    duration_months = date_range["duration_months"] if date_range else parse_explicit_duration_months(line)

    return {
        "raw": line,
        "company": company,
        "role": role,
        "start_date": date_range["start_date"] if date_range else None,
        "end_date": date_range["end_date"] if date_range else None,
        "duration_months": duration_months,
        "responsibilities": [],
        "technologies": [skill["name"] for skill in extract_skills(line)],
    }


def _parse_stacked_experience_header(company: str, role: str, date_line: str) -> dict:
    date_range = extract_date_range(date_line)
    return {
        "raw": f"{company} | {role} | {date_line}",
        "company": company,
        "role": role,
        "start_date": date_range["start_date"] if date_range else None,
        "end_date": date_range["end_date"] if date_range else None,
        "duration_months": date_range["duration_months"] if date_range else None,
        "responsibilities": [],
        "technologies": [],
    }


def _parse_company_role_pipe_date(company: str, role_date_line: str) -> dict | None:
    if "|" not in role_date_line:
        return None

    role_part, date_part = [part.strip(" -|,") for part in role_date_line.split("|", 1)]
    if not role_part or not _looks_like_role(role_part):
        return None

    date_range = extract_date_range(date_part)
    if not date_range:
        return None

    return {
        "raw": f"{company} | {role_part} | {date_part}",
        "company": company,
        "role": role_part,
        "start_date": date_range["start_date"],
        "end_date": date_range["end_date"],
        "duration_months": date_range["duration_months"],
        "responsibilities": [],
        "technologies": [skill["name"] for skill in extract_skills(role_part)],
    }


def _is_weak_role_fragment(line: str) -> bool:
    if not _looks_like_role(line):
        return False
    if extract_date_range(line) or parse_explicit_duration_months(line):
        return False
    if " at " in line.lower() or re.search(r"\s+-\s+|\s+\|\s+", line):
        return False
    return len(line.split()) <= 4


def _is_new_entry(line: str, *, has_current: bool = False) -> bool:
    if _normalize(line).startswith(PROJECT_META_PREFIXES):
        return False
    if extract_date_range(line) and (_looks_like_role(line) or " at " in line.lower()):
        return True
    if parse_explicit_duration_months(line) and _looks_like_role(line):
        return True
    if " at " in line.lower() and _looks_like_role(line):
        return True
    if re.search(r"\s+-\s+|\s+\|\s+", line) and _looks_like_role(line):
        return True

    if has_current and _is_weak_role_fragment(line):
        return False

    return _looks_like_role(line) and len(line.split()) <= 8


def _clean_lines(text: str) -> list[str]:
    return [line for line in (strip_list_marker(raw) for raw in (text or "").splitlines()) if line]


def _append_current(entries: list[dict], current: dict | None) -> None:
    if current:
        current["responsibilities"] = _merge_wrapped_responsibilities(current.get("responsibilities", []))
        current["technologies"] = _dedupe(current["technologies"])
        entries.append(current)


def _find_stacked_date_index_after_role(lines: list[str], company_index: int, *, max_lookahead: int = 9) -> int | None:
    if company_index + 1 >= len(lines):
        return None

    company = lines[company_index]
    role = lines[company_index + 1]
    if not (_looks_like_company(company) and _looks_like_role(role)):
        return None

    limit = min(len(lines), company_index + max_lookahead + 1)
    for date_index in range(company_index + 2, limit):
        candidate = lines[date_index]
        if _is_stop_line(candidate):
            break
        if _looks_like_company(candidate) and date_index > company_index + 2 and date_index + 1 < len(lines) and _looks_like_role(lines[date_index + 1]):
            break
        if _looks_like_project_block_start(candidate, lines[date_index + 1 : date_index + 6]):
            break
        if extract_date_range(candidate):
            return date_index

    return None


def _find_role_index_after_date(lines: list[str], company_index: int, *, max_lookahead: int = 6) -> tuple[int, int] | None:
    if company_index + 2 >= len(lines) or not _looks_like_company(lines[company_index]):
        return None

    limit = min(len(lines), company_index + max_lookahead + 1)
    for date_index in range(company_index + 1, limit):
        date_line = lines[date_index]
        if _is_stop_line(date_line):
            break
        if not extract_date_range(date_line):
            continue

        for role_index in range(date_index + 1, limit):
            role_line = lines[role_index]
            if _is_stop_line(role_line):
                break
            if _looks_like_role(role_line):
                return date_index, role_index

    return None


def _find_company_before_role(lines: list[str], role_index: int) -> int | None:
    if role_index <= 0 or not _looks_like_role(lines[role_index]):
        return None

    previous = lines[role_index - 1]
    if _looks_like_company(previous):
        return role_index - 1

    return None


def extract_experience(text: str) -> list[dict]:
    entries = []
    current = None
    lines = _clean_lines(text)
    index = 0

    while index < len(lines):
        line = lines[index]
        next_lines = lines[index + 1 : index + 6]

        if _is_stop_line(line) or _looks_like_project_block_start(line, next_lines):
            _append_current(entries, current)
            current = None
            break

        if _is_achievement_heading(line):
            index += 1
            continue

        if index + 1 < len(lines) and _looks_like_company(line):
            company_role_date = _parse_company_role_pipe_date(line, lines[index + 1])
            if company_role_date is not None:
                _append_current(entries, current)
                current = company_role_date
                index += 2
                continue

        date_role_indices = _find_role_index_after_date(lines, index)
        if date_role_indices is not None:
            date_index, role_index = date_role_indices
            _append_current(entries, current)
            current = _parse_stacked_experience_header(line, lines[role_index], lines[date_index])

            for detail in lines[index + 1 : date_index] + lines[date_index + 1 : role_index]:
                if _is_achievement_heading(detail):
                    continue
                current["responsibilities"].append(detail)
                current["technologies"].extend(skill["name"] for skill in extract_skills(detail))

            index = role_index + 1
            continue

        stacked_date_index = _find_stacked_date_index_after_role(lines, index)
        if stacked_date_index is not None:
            _append_current(entries, current)
            current = _parse_stacked_experience_header(line, lines[index + 1], lines[stacked_date_index])

            for detail in lines[index + 2 : stacked_date_index]:
                if _is_achievement_heading(detail):
                    continue
                current["responsibilities"].append(detail)
                current["technologies"].extend(skill["name"] for skill in extract_skills(detail))

            index = stacked_date_index + 1
            continue

        company_index = _find_company_before_role(lines, index)
        if company_index is not None:
            stacked_date_index = _find_stacked_date_index_after_role(lines, company_index)
            if stacked_date_index is not None:
                _append_current(entries, current)
                current = _parse_stacked_experience_header(
                    lines[company_index],
                    lines[index],
                    lines[stacked_date_index],
                )
                for detail in lines[index + 1 : stacked_date_index]:
                    if _is_achievement_heading(detail):
                        continue
                    current["responsibilities"].append(detail)
                    current["technologies"].extend(skill["name"] for skill in extract_skills(detail))
                index = stacked_date_index + 1
                continue

        if _is_new_entry(line, has_current=current is not None):
            _append_current(entries, current)
            current = _parse_experience_line(line)
            index += 1
            continue

        if current:
            current["responsibilities"].append(line)
            current["technologies"].extend(skill["name"] for skill in extract_skills(line))

        index += 1

    _append_current(entries, current)
    return entries
