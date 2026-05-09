import re

from app.parsers.normalizer import strip_accents, strip_list_marker


ACHIEVEMENT_KEYWORDS = (
    "achiev",
    "award",
    "bronze",
    "competition",
    "contest",
    "gold",
    "honor",
    "honour",
    "improved",
    "increased",
    "medal",
    "optimized",
    "prize",
    "ranked",
    "reduced",
    "second prize",
    "silver",
    "third prize",
    "won",
    "thanh tich",
    "giai",
)

HEADING_ONLY_ACHIEVEMENTS = {
    "achievement",
    "achievements",
    "key achievement",
    "key achievements",
    "achievement skills and knowledge gained",
    "achievements skills and knowledge gained",
}


def _normalize(value: str) -> str:
    normalized = strip_accents(value or "").lower()
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_heading_only(line: str) -> bool:
    return _normalize(line) in HEADING_ONLY_ACHIEVEMENTS


def extract_achievements(text: str, *, require_keyword: bool = False) -> list[dict]:
    achievements = []
    seen_titles = set()

    for raw in (text or "").splitlines():
        line = strip_list_marker(raw)
        if not line or _is_heading_only(line):
            continue

        lowered = strip_accents(line).lower()
        if require_keyword and not any(keyword in lowered for keyword in ACHIEVEMENT_KEYWORDS):
            continue

        normalized_title = _normalize(line)
        if normalized_title in seen_titles:
            continue

        seen_titles.add(normalized_title)
        year_match = re.search(r"(?:19|20)\d{2}", line)
        achievements.append(
            {
                "title": line,
                "description": None,
                "year": int(year_match.group(0)) if year_match else None,
            }
        )

    return sorted(achievements, key=lambda item: item["year"] is None)
