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
    "hoc tap",
    "truong",
    "dai hoc",
    "hoc vien",
    "sinh vien",
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

PRESENT_WORDS = {"nay", "present", "current", "now", "hien tai", "hiện tại"}

def _extract_years(text: str) -> tuple[int | None, int | None]:
    normalized = strip_accents(text or "").lower()
    years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", normalized)]

    if not years:
        return None, None

    start_year = years[0]
    end_year = years[1] if len(years) > 1 else None

    return start_year, end_year

def _extract_degree(text: str) -> str | None:
    normalized = strip_accents(text).lower()
    for pattern, degree in DEGREE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return degree

    if "sinh vien" in normalized or "dai hoc" in normalized:
        return "Bachelor"

    return None

def _extract_field(text: str) -> str | None:
    lowered = strip_accents(text).lower()
    for field, display_name in FIELD_PATTERNS.items():
        if field in lowered:
            return display_name
    return None

def _clean_institution_candidate(value: str) -> str:
    cleaned = value or ""

    cleaned = re.sub(
        r"^\s*(?:19|20)\d{2}\s*(?:-|–|—|to|den|đến)\s*"
        r"(?:nay|present|current|now|hien tai|hiện tại|(?:19|20)\d{2})\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r",?\s*(?:19|20)\d{2}\s*(?:-|–|—|to|den|đến)\s*(?:19|20)\d{2}\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    degree_split = re.split(
        r"\s+-\s+(?=(?:Bachelor|Master|PhD|Associate|B\.S\.|M\.S\.|Degree)\b)",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )

    if degree_split:
        cleaned = degree_split[0]

    return cleaned.strip(" -–—|,")

def _extract_institution(text: str) -> str | None:
    lines = [strip_list_marker(line) for line in (text or "").splitlines() if strip_list_marker(line)]

    for line in lines:
        normalized = strip_accents(line).lower()

        if any(keyword in normalized for keyword in ["university", "college", "institute", "academy", "school", "truong", "dai hoc", "hoc vien"]):
            return _clean_institution_candidate(line)

    return None

def _extract_gpa(text: str) -> str | None:
    match = re.search(r"\bGPA\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?\s*/\s*[0-9]+(?:\.[0-9]+)?)", text or "", re.IGNORECASE)
    return match.group(1).replace(" ", "") if match else None

def _looks_like_education_start(line: str) -> bool:
    normalized = strip_accents(line).lower()

    education_keywords = [
        "dai hoc",
        "university",
        "college",
        "institute",
        "academy",
        "school",
        "hoc vien",
        "truong",
    ]

    has_school_keyword = any(keyword in normalized for keyword in education_keywords)
    has_year = bool(re.search(r"(?:19|20)\d{2}", normalized))

    if has_school_keyword:
        return True

    if has_year and any(keyword in normalized for keyword in ["bachelor", "master", "phd", "degree"]):
        return True

    return False

def _split_education_blocks(text: str) -> list[list[str]]:
    lines = [strip_list_marker(raw) for raw in (text or "").splitlines()]
    lines = _merge_wrapped_lines([line for line in lines if line])

    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if _looks_like_education_start(line):
            if current:
                blocks.append(current)
            current = [line]
            continue

        if current:
            current.append(line)

    if current:
        blocks.append(current)

    return blocks

def extract_education(text: str) -> list[dict]:
    results = []

    blocks = _split_education_blocks(text)

    if not blocks:
        candidate_lines = []
        for raw in (text or "").splitlines():
            line = strip_list_marker(raw)
            if not line:
                continue

            lowered = strip_accents(line).lower()
            if any(keyword in lowered for keyword in DEGREE_KEYWORDS):
                candidate_lines.append(line)

        if candidate_lines:
            blocks = [candidate_lines]

    for block in blocks:
        block_text = "\n".join(block)
        start_year, end_year = _extract_years(block_text)
        gpa = _extract_gpa(block_text)

        description = block_text

        if gpa and "gpa" not in block_text.lower():
            description = f"{description}\nGPA: {gpa}"

        results.append(
            {
                "institution": _extract_institution(block_text),
                "degree": _extract_degree(block_text),
                "field_of_study": _extract_field(block_text),
                "start_year": start_year,
                "end_year": end_year,
                "description": description,
            }
        )

    return results

def _merge_wrapped_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []

    for line in lines:
        clean = line.strip()
        if not clean:
            continue

        if merged and (
            merged[-1].endswith("(")
            or merged[-1].count("(") > merged[-1].count(")")
            or len(merged[-1].split()) <= 3
        ):
            merged[-1] = f"{merged[-1]} {clean}"
            continue

        merged.append(clean)

    return merged