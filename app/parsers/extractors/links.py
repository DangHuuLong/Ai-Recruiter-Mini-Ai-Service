import re
from urllib.parse import urlsplit, urlunsplit


URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com|(?:[\w-]+\.)+[a-z]{2,})(?:/[\w\-./?=#%&+]*)?",
    re.IGNORECASE,
)

GITHUB_PROFILE_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<username>[A-Za-z0-9-]+)(?:/.*)?$",
    re.IGNORECASE,
)

GITHUB_RESERVED_PATHS = {"repositories", "projects", "stars", "followers", "following"}


def _canonical_key(value: str) -> str:
    return value.rstrip("/").lower()


def _unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = _canonical_key(value)
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


def _normalize_github_profile_url(url: str) -> str | None:
    match = GITHUB_PROFILE_RE.match(url)
    if not match:
        return None

    username = match.group("username")
    if username.lower() in GITHUB_RESERVED_PATHS:
        return None

    return f"https://github.com/{username}"


def _with_trailing_slash_for_domain_only(url: str) -> str:
    parts = urlsplit(url)
    if parts.path in {"", "/"} and not parts.query and not parts.fragment:
        return urlunsplit((parts.scheme, parts.netloc, "/", "", ""))
    return url


def _looks_like_portfolio_url(url: str) -> bool:
    lowered = url.lower()

    if "linkedin.com" in lowered or "github.com" in lowered:
        return False

    return any(token in lowered for token in ["portfolio", "behance", "dribbble", "github.io", "super.site"]) or re.search(
        r"\.(dev|me|io|app)(/|$)", lowered
    ) is not None


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
            if result["linkedin"] is None:
                result["linkedin"] = url
        elif "github.com" in lowered:
            github_profile_url = _normalize_github_profile_url(url)
            if github_profile_url and result["github"] is None:
                result["github"] = github_profile_url
        elif _looks_like_portfolio_url(url):
            if result["portfolio"] is None:
                result["portfolio"] = _with_trailing_slash_for_domain_only(url)
        else:
            result["other"].append(url)

    return result
