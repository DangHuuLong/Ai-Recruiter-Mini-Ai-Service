import re
from collections.abc import Iterable

from app.parsers.skill_catalog import SKILL_CATALOG


def _alias_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias)

    if alias == ".net":
        return re.compile(rf"(?<![\w+]){escaped}(?![\w+])", re.IGNORECASE)

    # Keep short aliases strict so they do not fire on framework suffixes
    # such as Node.js, Express.js, Next.js, or file extensions.
    if alias in {"js", "ts"}:
        return re.compile(rf"(?<![.\w]){escaped}(?![\w])", re.IGNORECASE)

    # Plain C should only match standalone C, C/C++, or C, C++ style language lists.
    # This avoids false positives in words like CSS, Cloudinary, React, or academic.
    if alias == "c":
        return re.compile(r"(?<![A-Za-z0-9+#.])c(?![A-Za-z0-9#.])", re.IGNORECASE)

    # Do not extract plain React from React Native; React Native has its own
    # catalog entry and should remain the more specific match.
    if alias == "react":
        return re.compile(rf"\b{escaped}\b(?!\s+native)", re.IGNORECASE)

    # Do not extract plain CSS from Tailwind CSS; Tailwind CSS is the explicit
    # technology in that phrase.
    if alias == "css":
        return re.compile(rf"(?<!tailwind\s)\b{escaped}\b", re.IGNORECASE)

    if alias in {
        "c#", "c++", "ci/cd", "shadcn/ui", "socket.io", "bloc/cubit",
        "tcp/ip", "ids/ips", "ssl/tls", "f#", "q#",
    }:
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)

    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def _iter_evidence_units(text: str) -> Iterable[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if lines:
        yield from lines

        # PDF extraction can wrap one logical skill line across multiple physical lines,
        # e.g. "TanStack" on one line and "Query" on the next. Search small windows too.
        for index in range(len(lines) - 1):
            yield f"{lines[index]} {lines[index + 1]}"
        for index in range(len(lines) - 2):
            yield f"{lines[index]} {lines[index + 1]} {lines[index + 2]}"
        return

    yield from (part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip())


def extract_skills(text: str) -> list[dict[str, str | None]]:
    """Extract known technical skills from resume text."""
    found: dict[str, dict[str, str | None]] = {}

    for evidence in _iter_evidence_units(text):
        for definition in SKILL_CATALOG:
            if definition.name in found:
                continue

            if any(_alias_pattern(alias).search(evidence) for alias in definition.aliases):
                found[definition.name] = {
                    "name": definition.name,
                    "normalized_name": definition.normalized_name,
                    "category": definition.category,
                    "evidence": evidence,
                }

    return list(found.values())
