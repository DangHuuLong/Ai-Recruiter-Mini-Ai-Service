import re

from app.parsers.extractors.skills import extract_skills
from app.parsers.normalizer import strip_accents, strip_list_marker


URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}(?:/[\w\-./?=#%&+]*)?",
    re.IGNORECASE,
)

TECH_AS_URL_BLACKLIST = {
    "node.js",
    "react.js",
    "next.js",
    "vue.js",
    "express.js",
    "nest.js",
    "angular.js",
}

FIELD_LABELS = {
    "technologies": (
        "technology",
        "technologies",
        "tech",
        "tech stack",
        "tools",
        "environment",
        "stack",
        "cong nghe",
        "cong nghe su dung",
        "ky thuat su dung",
        "nen tang su dung",
    ),
    "description": (
        "description",
        "features",
        "feature",
        "main features",
        "function",
        "functions",
        "functionality",
        "mo ta",
        "mo ta chuc nang",
        "chuc nang",
        "tinh nang",
    ),
    "role": (
        "role",
        "my role",
        "responsibility",
        "responsibilities",
        "vai tro",
        "phu trach",
        "nhiem vu",
    ),
    "status": (
        "status",
        "state",
        "trang thai",
        "tinh trang",
    ),
    "link": (
        "link",
        "links",
        "github",
        "gitlab",
        "demo",
        "source",
        "repository",
        "repo",
        "website",
    ),
    "meta": (
        "team",
        "team size",
        "members",
        "member",
        "du an nhom",
        "du an ca nhan",
        "personal project",
        "group project",
    ),
}

FEATURE_STARTERS = (
    "crud",
    "upload",
    "thanh toan",
    "dang nhap",
    "dang ky",
    "quan ly tai khoan",
    "gio hang",
    "customer",
    "admin",
    "auth",
    "api ",
)

DATE_RANGE_RE = re.compile(
    r"(?:"
    r"(?:19|20)\d{2}(?:[/.-]\d{1,2})?"
    r"|"
    r"\d{1,2}[/.-](?:19|20)\d{2}"
    r")"
    r"\s*(?:-|–|—|to|den|đến)\s*"
    r"(?:nay|present|current|now|hien tai|hiện tại|(?:19|20)\d{2}(?:[/.-]\d{1,2})?|\d{1,2}[/.-](?:19|20)\d{2})",
    re.IGNORECASE,
)


def _normalize_key(value: str) -> str:
    text = strip_accents(value or "").lower()
    text = re.sub(r"[^a-z0-9+#./ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_date_suffix(value: str) -> str:
    date_match = DATE_RANGE_RE.search(value)
    if date_match:
        return value[: date_match.start()].strip(" -|,")

    date_start_match = re.search(r"\s+(?:19|20)\d{2}(?:[/.-]\d{1,2})?\b", value)
    if date_start_match:
        return value[: date_start_match.start()].strip(" -|,")

    return value.strip()


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _extract_url_match(line: str) -> re.Match[str] | None:
    return URL_RE.search(line or "")


def _extract_url(line: str) -> str | None:
    match = _extract_url_match(line)
    if not match:
        return None

    url = match.group(0).rstrip(".,;)")
    lowered = url.lower()

    if lowered in TECH_AS_URL_BLACKLIST:
        return None

    if url.startswith(("http://", "https://")):
        return url

    return f"https://{url}"


def _remove_url_text(line: str, url: str | None) -> str:
    if not url:
        return line.strip()

    raw_url = url.replace("https://", "").replace("http://", "")
    return line.replace(url, "").replace(raw_url, "").strip(" -|,")


def _is_url_only_line(line: str, url: str | None = None) -> bool:
    clean = line.strip()
    if not clean:
        return False

    found_url = url or _extract_url(clean)
    if not found_url:
        return False

    return _remove_url_text(clean, found_url) == ""


def _extract_year(line: str) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", line or "")
    return int(match.group(0)) if match else None


def _extract_issuer(line: str) -> str | None:
    parts = [part.strip(" -|,") for part in re.split(r"\s+-\s+|\s+\|\s+|,", line) if part.strip(" -|,")]
    if len(parts) >= 2:
        return parts[1]
    return None


def _split_label_value(line: str) -> tuple[str | None, str | None]:
    if ":" not in line:
        return None, None

    label, value = line.split(":", 1)
    label = label.strip()
    value = value.strip()

    if not label:
        return None, None

    return label, value


def _label_kind(line: str) -> str | None:
    label, _ = _split_label_value(line)
    candidate = label if label is not None else line
    normalized = _normalize_key(candidate)

    for kind, aliases in FIELD_LABELS.items():
        for alias in aliases:
            if normalized == _normalize_key(alias):
                return kind

    return None


def _is_field_label_line(line: str) -> bool:
    return _label_kind(line) is not None


def _looks_like_sentence(line: str) -> bool:
    clean = line.strip()
    if not clean:
        return False

    if clean.endswith((".", "!", "?")):
        return True

    return len(clean.split()) > 10


def _looks_like_project_title(line: str) -> bool:
    clean = _strip_date_suffix(strip_list_marker(line).strip())
    normalized = _normalize_key(clean)

    if not clean:
        return False

    if _is_field_label_line(clean):
        return False

    if clean.endswith(":"):
        return False

    if ":" in clean:
        return False

    if "|" in clean:
        return False

    if _looks_like_sentence(clean):
        return False

    if any(normalized.startswith(prefix) for prefix in FEATURE_STARTERS):
        return False

    word_count = len(clean.split())
    return 1 <= word_count <= 8


def _has_dated_project_title(line: str) -> bool:
    clean = strip_list_marker(line).strip()
    if not DATE_RANGE_RE.search(clean):
        return False

    title = _strip_date_suffix(clean)
    return title != clean and _looks_like_project_title(title)


def _has_strong_project_header_context(line: str) -> bool:
    normalized = _normalize_key(line)

    if DATE_RANGE_RE.search(line):
        return True

    if _is_url_only_line(line):
        return True

    strong_tokens = [
        "link github",
        "github",
        "gitlab",
        "demo",
        "repository",
        "repo",
        "du an nhom",
        "du an ca nhan",
        "personal project",
        "group project",
        "cong nghe su dung",
        "technologies",
        "tech stack",
        "position",
        "role",
        "teamsize",
        "description",
    ]

    if any(token in normalized for token in strong_tokens):
        return True

    if _label_kind(line) in {"technologies", "link", "meta", "role", "description"}:
        return True

    return False


def _should_start_new_block(line: str, current: list[str], next_lines: list[str]) -> bool:
    clean = strip_list_marker(line).strip()

    if not _looks_like_project_title(clean):
        return False

    if not current:
        return True

    if _has_dated_project_title(clean):
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


def _clean_description_parts(parts: list[str]) -> str | None:
    cleaned = []
    for part in parts:
        value = re.sub(r"\s+", " ", part or "").strip()
        if value:
            cleaned.append(value)

    return " ".join(cleaned) if cleaned else None


def _append_description(parts: list[str], label: str | None, value: str) -> None:
    value = value.strip()
    if not value:
        return

    if label:
        parts.append(f"{label}: {value}")
    else:
        parts.append(value)


def parse_project_block(lines: list[str]) -> dict:
    if not lines:
        return {
            "name": None,
            "description": None,
            "technologies": [],
            "url": None,
        }

    first_line = lines[0].strip()

    url = _extract_url(first_line)

    name, initial_description = _split_inline_project_header(first_line)
    description_parts: list[str] = []
    technologies: list[str] = []

    if initial_description:
        description_parts.append(initial_description)

    if url:
        raw_url = url.replace("https://", "").replace("http://", "")
        name = name.replace(url, "").replace(raw_url, "").strip(" -|,")

        description_parts = [
            part.replace(url, "").replace(raw_url, "").strip()
            for part in description_parts
            if part.replace(url, "").replace(raw_url, "").strip()
        ]

    for line in lines[1:]:
        clean = line.strip()
        if not clean:
            continue

        found_url = _extract_url(clean)
        if found_url and not url:
            url = found_url

        technologies.extend(skill["name"] for skill in extract_skills(clean))

        if found_url and _is_url_only_line(clean, found_url):
            continue

        if found_url:
            clean = _remove_url_text(clean, found_url)
            if not clean:
                continue

        label, value = _split_label_value(clean)
        kind = _label_kind(clean)

        if kind and not value:
            continue

        if kind == "technologies":
            technologies.extend(skill["name"] for skill in extract_skills(value or clean))
            _append_description(description_parts, label, value or clean)
            continue

        if kind == "link":
            continue

        if kind in {"description", "role", "status", "meta"}:
            _append_description(description_parts, label, value or clean)
            continue

        _append_description(description_parts, None, clean)

    technologies.extend(skill["name"] for skill in extract_skills(first_line))

    return {
        "name": name or None,
        "description": _clean_description_parts(description_parts),
        "technologies": _dedupe(technologies),
        "url": url,
    }


def _is_valid_project(project: dict) -> bool:
    name = project.get("name")
    if not name:
        return False

    normalized_name = _normalize_key(name)

    for aliases in FIELD_LABELS.values():
        if normalized_name in {_normalize_key(alias) for alias in aliases}:
            return False

    if len(str(name).split()) > 12:
        return False

    return True


def extract_projects(text: str) -> list[dict]:
    projects = []

    for block in split_project_blocks(text):
        project = parse_project_block(block)
        if _is_valid_project(project):
            projects.append(project)

    return projects


def extract_certifications(text: str, *, allow_year_only: bool = False) -> list[dict]:
    certs = []
    for raw in (text or "").splitlines():
        line = strip_list_marker(raw)
        if not line:
            continue

        lowered = line.lower()
        is_cert_line = any(keyword in lowered for keyword in ["certificat", "certified", "certificate", "certicate", "chung chi"])
        has_year = bool(_extract_year(line))

        if is_cert_line or (allow_year_only and has_year):
            url_match = _extract_url_match(line)
            name = line.replace(url_match.group(0), "") if url_match else line
            certs.append(
                {
                    "name": name.strip(" -|,"),
                    "issuer": _extract_issuer(name),
                    "issued_year": _extract_year(line),
                    "url": _extract_url(line),
                }
            )

    return certs


def _split_inline_project_header(line: str) -> tuple[str, str | None]:
    clean = line.strip()

    url_match = _extract_url_match(clean)
    if url_match:
        clean = clean.replace(url_match.group(0), "").strip()

    stripped_name = _strip_date_suffix(clean)
    if stripped_name != clean and stripped_name and len(stripped_name.split()) <= 8:
        return stripped_name, None

    for pattern in (r"\s+-\s+", r":\s+"):
        parts = re.split(pattern, clean, maxsplit=1)
        if len(parts) == 2:
            name = parts[0].strip(" -|,")
            description = parts[1].strip(" -|,")
            if name and description and len(name.split()) <= 8:
                return name, description

    return line.strip(" -|,"), None