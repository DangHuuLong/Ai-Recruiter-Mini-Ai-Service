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

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE)

GITHUB_RESERVED_PATHS = {"repositories", "projects", "stars", "followers", "following"}
COMMON_EMAIL_DOMAINS = {
    "gmail",
    "yahoo",
    "outlook",
    "hotmail",
    "icloud",
    "protonmail",
    "live",
}


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


def _github_username(url: str) -> str | None:
    match = GITHUB_PROFILE_RE.match(url)
    if not match:
        return None

    username = match.group("username")
    if username.lower() in GITHUB_RESERVED_PATHS:
        return None

    return username


def _normalize_github_profile_url(username: str) -> str:
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


def _candidate_identity_tokens(text: str, urls: list[str]) -> set[str]:
    tokens = set()

    for email_match in EMAIL_RE.finditer(text or ""):
        local_part = email_match.group(0).split("@", 1)[0]
        if local_part:
            tokens.add(local_part.lower())

    for url in urls:
        parts = urlsplit(url)
        host_parts = [part for part in parts.netloc.lower().split(".") if part]
        if not host_parts:
            continue

        first_label = host_parts[0]
        if first_label not in {"www", "github"} and first_label not in COMMON_EMAIL_DOMAINS:
            tokens.add(first_label)

    return tokens


def _select_github_profile_url(github_urls: list[str], identity_tokens: set[str]) -> str | None:
    usernames = []
    for url in github_urls:
        username = _github_username(url)
        if username:
            usernames.append(username)

    usernames = _unique_in_order(usernames)
    if not usernames:
        return None

    for username in usernames:
        if username.lower() in identity_tokens:
            return _normalize_github_profile_url(username)

    return _normalize_github_profile_url(usernames[0])


def extract_links(text: str) -> dict:
    urls = []
    for match in URL_RE.finditer(text or ""):
        if match.start() > 0 and text[match.start() - 1] == "@":
            continue
        urls.append(_normalize_url(match.group(0)))

    urls = _unique_in_order(urls)
    identity_tokens = _candidate_identity_tokens(text or "", urls)
    github_urls = [url for url in urls if "github.com" in url.lower()]

    result = {"linkedin": None, "github": _select_github_profile_url(github_urls, identity_tokens), "portfolio": None, "other": []}
    for url in urls:
        lowered = url.lower()
        if "linkedin.com" in lowered:
            if result["linkedin"] is None:
                result["linkedin"] = url
        elif "github.com" in lowered:
            continue
        elif _looks_like_portfolio_url(url):
            if result["portfolio"] is None:
                result["portfolio"] = _with_trailing_slash_for_domain_only(url)
        else:
            result["other"].append(url)

    return result
