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
    "hcmus",
]

DEGREE_PATTERNS = [
    (r"\b(bachelor|bachelor'?s|b\.?s\.?|bsc|b\.?eng|engineer)\b", "Bachelor"),
    (r"\b(cu nhan|ky su)\b", "Bachelor"),
    (r"\b(master|master'?s|m\.?s\.?|msc|mba)\b", "Master"),
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
    "it": "Information Technology",
}

KNOWN_INSTITUTIONS = {
    "hcmus": "HCMUS",
}

PRESENT_WORDS = {"nay", "present", "current", "now", "hien tai", "hiện tại"}

DATE_RANGE_RE = re.compile(
    r"(?:"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+"
    r")?"
    r"(?:19|20)\d{2}"
    r"\s*(?:-|–|—|to|den|đến)\s*"
    r"(?:nay|present|current|now|hien tai|hiện tại|"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+)?(?:19|20)\d{2})",
    re.IGNORECASE,
)


def _extract_years(text: str) -> tuple[int | None, int | None]:
    normalized = strip_accents(text or "").lower()
    years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", normalized)]

    if not years:
        return None, None

    start_year = years[0]
    end_year = years[1] if len(years) > 1 and not any(word in normalized for word in PRESENT_WORDS) else None

    return start_year, end_year


def _is_date_line(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if not clean:
        return False

    if DATE_RANGE_RE.search(clean):
        return True

    normalized = strip_accents(clean).lower()
    return bool(
        re.fullmatch(
            r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+)?(?:19|20)\d{2}",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _normalize_degree_text(text: str) -> str:
    return (
        strip_accents(text or "")
        .lower()
        .replace("ˇ", "'")
        .replace("’", "'")
        .replace("`", "'")
        .replace("´", "'")
    )


def _extract_degree(text: str) -> str | None:
    normalized = _normalize_degree_text(text)
    for pattern, degree in DEGREE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return degree

    if "sinh vien" in normalized or "dai hoc" in normalized or "hcmus" in normalized:
        return "Bachelor"

    return None


def _extract_field(text: str) -> str | None:
    lowered = strip_accents(text).lower()
    for field, display_name in FIELD_PATTERNS.items():
        if re.search(rf"\b{re.escape(field)}\b", lowered):
            return display_name
    return None


def _clean_institution_candidate(value: str) -> str:
    cleaned = value or ""

    cleaned = DATE_RANGE_RE.sub("", cleaned)

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


def _matches_known_institution(normalized: str, token: str) -> bool:
    return normalized == token or normalized.startswith(f"{token} ")


def _is_school_line(line: str) -> bool:
    normalized = strip_accents(line).lower()
    return any(keyword in normalized for keyword in ["university", "college", "institute", "academy", "school", "truong", "dai hoc", "hoc vien"])


def _extract_institution(text: str) -> str | None:
    lines = [strip_list_marker(line) for line in (text or "").splitlines() if strip_list_marker(line)]
    institution_lines: list[str] = []

    for line in lines:
        normalized = strip_accents(line).lower()

        if _is_date_line(line):
            continue

        for token, display_name in KNOWN_INSTITUTIONS.items():
            if _matches_known_institution(normalized, token):
                return display_name

        if _is_school_line(line):
            institution = _clean_institution_candidate(line)
            if institution:
                institution_lines.append(institution)
            continue

        if institution_lines and not _extract_degree(line) and not _extract_field(line) and not _extract_years(line)[0] and "gpa" not in normalized:
            institution = _clean_institution_candidate(line)
            if institution:
                institution_lines.append(institution)

    if institution_lines:
        return " - ".join(line for line in institution_lines if line).strip(" -–—|,") or None

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
        "hcmus",
    ]

    has_school_keyword = any(keyword in normalized for keyword in education_keywords)
    has_year = bool(re.search(r"(?:19|20)\d{2}", normalized))

    if has_school_keyword:
        return True

    if has_year and any(keyword in normalized for keyword in ["bachelor", "master", "phd", "degree"]):
        return True

    return False


def _looks_like_same_education_item(current: list[str], line: str) -> bool:
    if not current:
        return False

    normalized = strip_accents(line).lower()
    current_text = "\n".join(current)
    current_has_school = any(_is_school_line(item) for item in current)
    current_has_degree = _extract_degree(current_text) is not None

    if _is_date_line(line):
        return True

    if current[-1].rstrip().endswith(("-", "–", "—")):
        return True

    if _is_school_line(line) and current_has_school and not current_has_degree:
        return True

    return "gpa" in normalized or _extract_field(line) is not None or _extract_degree(line) is not None


def _merge_date_only_blocks(blocks: list[list[str]]) -> list[list[str]]:
    merged: list[list[str]] = []
    pending_date_block: list[str] | None = None

    for block in blocks:
        if block and all(_is_date_line(line) for line in block):
            pending_date_block = block
            continue

        if pending_date_block:
            block = [*pending_date_block, *block]
            pending_date_block = None

        merged.append(block)

    if pending_date_block:
        merged.append(pending_date_block)

    return merged


def _split_education_blocks(text: str) -> list[list[str]]:
    lines = [strip_list_marker(raw) for raw in (text or "").splitlines()]
    lines = _merge_wrapped_lines([line for line in lines if line])

    blocks: list[list[str]] = []
    current: list[str] = []

    for index, line in enumerate(lines):
        if not current and _is_date_line(line):
            next_lines = lines[index + 1 : index + 4]
            if any(_looks_like_education_start(next_line) for next_line in next_lines):
                current = [line]
            continue

        if _looks_like_education_start(line):
            if current and not _looks_like_same_education_item(current, line):
                blocks.append(current)
                current = [line]
                continue

            current.append(line)
            continue

        if current:
            current.append(line)

    if current:
        blocks.append(current)

    return _merge_date_only_blocks(blocks)


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
            if any(keyword in lowered for keyword in DEGREE_KEYWORDS) or _is_date_line(line):
                candidate_lines.append(line)

        if candidate_lines:
            blocks = _merge_date_only_blocks([candidate_lines])

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

        if merged and merged[-1].rstrip().endswith(("-", "–", "—")):
            if _is_date_line(clean):
                merged.append(clean)
            else:
                merged[-1] = f"{merged[-1].rstrip(' -–—')} - {clean}"
            continue

        if merged and not _is_date_line(merged[-1]) and (
            merged[-1].endswith("(")
            or merged[-1].count("(") > merged[-1].count(")")
            or (len(merged[-1].split()) <= 3 and not _is_date_line(clean))
        ):
            merged[-1] = f"{merged[-1]} {clean}"
            continue

        merged.append(clean)

    return merged
