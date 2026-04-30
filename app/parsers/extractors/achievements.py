import re

from app.parsers.normalizer import strip_list_marker


ACHIEVEMENT_KEYWORDS = (
    "achiev",
    "award",
    "honor",
    "improved",
    "increased",
    "reduced",
    "optimized",
    "won",
    "ranked",
    "thanh tich",
    "giai",
)


def extract_achievements(text: str, *, require_keyword: bool = False) -> list[dict]:
    achievements = []

    for raw in (text or "").splitlines():
        line = strip_list_marker(raw)
        if not line:
            continue

        lowered = line.lower()
        if require_keyword and not any(keyword in lowered for keyword in ACHIEVEMENT_KEYWORDS):
            continue

        year_match = re.search(r"(?:19|20)\d{2}", line)
        achievements.append(
            {
                "title": line,
                "description": None,
                "year": int(year_match.group(0)) if year_match else None,
            }
        )

    return sorted(achievements, key=lambda item: item["year"] is None)
