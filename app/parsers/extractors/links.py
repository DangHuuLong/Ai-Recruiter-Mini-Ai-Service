import re


URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com|[\w-]+\.(?:app|dev|io|me|net|org|com))(?:/[\w\-./?=#%&+]*)?",
    re.IGNORECASE,
)


def _unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _normalize_url(url: str) -> str:
    cleaned = url.rstrip(".,;)")
    if cleaned.startswith(("http://", "https://")):
        return cleaned
    return f"https://{cleaned}"


def extract_links(text: str) -> dict:
    urls = []
    for match in URL_RE.finditer(text or ""):
        if match.start() > 0 and text[match.start() - 1] == "@":
            continue
        urls.append(_normalize_url(match.group(0)))

    result = {"linkedin": None, "github": None, "portfolio": None, "other": []}
    for url in _unique_in_order(urls):
        lowered = url.lower()
        if "linkedin.com" in lowered:
            result["linkedin"] = url
        elif "github.com" in lowered:
            result["github"] = url
        elif any(token in lowered for token in ["portfolio", "behance", "dribbble"]) or re.search(
            r"\.(dev|me|io|app)(/|$)", lowered
        ):
            result["portfolio"] = url
        else:
            result["other"].append(url)

    return result
