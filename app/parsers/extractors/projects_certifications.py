import re

from app.parsers.extractors.skills import extract_skills
from app.parsers.normalizer import strip_list_marker


URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}(?:/[\w\-./?=#%&+]*)?", re.IGNORECASE)


def _extract_url_match(line: str) -> re.Match[str] | None:
    return URL_RE.search(line)


def _extract_url(line: str) -> str | None:
    match = URL_RE.search(line)
    if not match:
        return None

    url = match.group(0).rstrip(".,;)")
    if url.startswith(("http://", "https://")):
        return url
    return f"https://{url}"


def _extract_year(line: str) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", line)
    return int(match.group(0)) if match else None


def _extract_issuer(line: str) -> str | None:
    parts = [part.strip(" -|,") for part in re.split(r"\s+-\s+|\s+\|\s+|,", line) if part.strip(" -|,")]
    if len(parts) >= 2:
        return parts[1]
    return None


def _project_from_line(line: str) -> dict:
    url_match = _extract_url_match(line)
    url = _extract_url(line)
    text_without_url = line
    if url_match:
        text_without_url = line.replace(url_match.group(0), "")
    text_without_url = text_without_url.strip(" -|,")
    parts = [part.strip() for part in re.split(r"\s+-\s+|:\s+", text_without_url, maxsplit=1) if part.strip()]

    return {
        "name": parts[0] if parts else text_without_url,
        "description": parts[1] if len(parts) > 1 else None,
        "technologies": [skill["name"] for skill in extract_skills(line)],
        "url": url,
    }


def extract_projects(text: str) -> list[dict]:
    projects = []
    current = None

    for raw in (text or "").splitlines():
        line = strip_list_marker(raw)
        if not line:
            continue

        if current is None or not raw.lstrip().startswith(("-", "*")):
            if current:
                current["technologies"] = list(dict.fromkeys(current["technologies"]))
                projects.append(current)
            current = _project_from_line(line)
            continue

        current["description"] = " ".join(part for part in [current.get("description"), line] if part)
        current["technologies"].extend(skill["name"] for skill in extract_skills(line))

    if current:
        current["technologies"] = list(dict.fromkeys(current["technologies"]))
        projects.append(current)

    return projects


def extract_certifications(text: str) -> list[dict]:
    certs = []
    for raw in (text or "").splitlines():
        line = strip_list_marker(raw)
        if not line:
            continue

        lowered = line.lower()
        if any(keyword in lowered for keyword in ["certificat", "certified", "certificate", "chung chi"]):
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
