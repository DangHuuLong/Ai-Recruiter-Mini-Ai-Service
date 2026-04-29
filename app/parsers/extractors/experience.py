import re

from app.parsers.extractors.skills import extract_skills
from app.parsers.normalizer import strip_list_marker
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


def _looks_like_role(value: str) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in ROLE_KEYWORDS)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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


def _is_new_entry(line: str) -> bool:
    if extract_date_range(line) and (_looks_like_role(line) or " at " in line.lower()):
        return True
    if parse_explicit_duration_months(line) and _looks_like_role(line):
        return True
    if " at " in line.lower() and _looks_like_role(line):
        return True
    if re.search(r"\s+-\s+|\s+\|\s+", line) and _looks_like_role(line):
        return True
    return _looks_like_role(line) and len(line.split()) <= 8


def extract_experience(text: str) -> list[dict]:
    entries = []
    current = None

    for raw in (text or "").splitlines():
        line = strip_list_marker(raw)
        if not line:
            continue

        if _is_new_entry(line):
            if current:
                current["technologies"] = _dedupe(current["technologies"])
                entries.append(current)
            current = _parse_experience_line(line)
            continue

        if current:
            current["responsibilities"].append(line)
            current["technologies"].extend(skill["name"] for skill in extract_skills(line))

    if current:
        current["technologies"] = _dedupe(current["technologies"])
        entries.append(current)

    return entries
