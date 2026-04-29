import re

from app.parsers.normalizer import strip_accents, strip_list_marker


DEGREE_KEYWORDS = [
    "university",
    "college",
    "institute",
    "academy",
    "school",
    "bachelor",
    "master",
    "phd",
    "degree",
    "hoc van",
    "truong",
    "dai hoc",
    "hoc vien",
]

DEGREE_PATTERNS = [
    (r"\b(bachelor|b\.?s\.?|bsc|b\.?eng|engineer)\b", "Bachelor"),
    (r"\b(cu nhan|ky su)\b", "Bachelor"),
    (r"\b(master|m\.?s\.?|msc|mba)\b", "Master"),
    (r"\b(thac si|thac sy)\b", "Master"),
    (r"\b(phd|ph\.?d|doctor)\b", "PhD"),
    (r"\b(tien si)\b", "PhD"),
    (r"\b(associate)\b", "Associate"),
]

FIELD_PATTERNS = {
    "computer science": "Computer Science",
    "khoa hoc may tinh": "Computer Science",
    "software engineering": "Software Engineering",
    "ky thuat phan mem": "Software Engineering",
    "information technology": "Information Technology",
    "cong nghe thong tin": "Information Technology",
    "data science": "Data Science",
    "computer engineering": "Computer Engineering",
    "business administration": "Business Administration",
}


def _extract_years(line: str) -> tuple[int | None, int | None]:
    years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", line)]
    if not years:
        return None, None
    if len(years) == 1:
        return years[0], None
    return years[0], years[1]


def _extract_degree(line: str) -> str | None:
    normalized = strip_accents(line).lower()
    for pattern, degree in DEGREE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return degree
    return None


def _extract_field(line: str) -> str | None:
    lowered = strip_accents(line).lower()
    for field, display_name in FIELD_PATTERNS.items():
        if field in lowered:
            return display_name
    return None


def _extract_institution(line: str) -> str | None:
    parts = [part.strip(" -|,") for part in re.split(r"\s+-\s+|\s+\|\s+|,", line) if part.strip(" -|,")]
    for part in parts:
        lowered = strip_accents(part).lower()
        if any(
            keyword in lowered
            for keyword in ["university", "college", "institute", "academy", "school", "truong", "dai hoc", "hoc vien"]
        ):
            return part

    if parts and not _extract_degree(parts[0]):
        return parts[0]

    return None


def extract_education(text: str) -> list[dict]:
    results = []
    for raw in (text or "").splitlines():
        line = strip_list_marker(raw)
        if not line:
            continue

        lowered = strip_accents(line).lower()
        if not any(keyword in lowered for keyword in DEGREE_KEYWORDS):
            continue

        start_year, end_year = _extract_years(line)
        results.append(
            {
                "institution": _extract_institution(line),
                "degree": _extract_degree(line),
                "field_of_study": _extract_field(line),
                "start_year": start_year,
                "end_year": end_year,
                "description": line,
            }
        )

    return results
