import re
from datetime import datetime
from typing import Optional

from app.parsers.normalizer import strip_accents


PRESENT_VALUES = {"present", "current", "now", "to date", "nay", "hien tai"}
MONTH_FORMATS = (
    "%b %Y",
    "%B %Y",
    "%b. %Y",
    "%B. %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d %b. %Y",
    "%d %B. %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%b. %d %Y",
    "%B. %d %Y",
    "%Y %b",
    "%Y %B",
    "%Y %b.",
    "%Y %B.",
)


def _normalize_date_text(value: str) -> str:
    normalized = value or ""
    normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
    normalized = normalized.replace("〜", "-").replace("~", "-")
    normalized = re.sub(r"\b(?:den|đến)\b", " to ", normalized, flags=re.IGNORECASE)
    return normalized


def _clean_date_value(value: str) -> str:
    cleaned = strip_accents(_normalize_date_text(value)).replace(",", " ").strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned.startswith("sept "):
        cleaned = cleaned.replace("sept ", "sep ", 1)
    return cleaned


def _parse_year_month(value: str) -> Optional[datetime]:
    value = _clean_date_value(value)
    if not value:
        return None

    if value in PRESENT_VALUES:
        return datetime.today().replace(day=1)

    try:
        year_month_day = re.fullmatch(
            r"(?P<year>(?:19|20)\d{2})[-/.](?P<month>\d{1,2})(?:[-/.](?P<day>\d{1,2}))?",
            value,
        )
        if year_month_day:
            return datetime(
                int(year_month_day.group("year")),
                int(year_month_day.group("month")),
                int(year_month_day.group("day") or 1),
            )

        day_month_year = re.fullmatch(
            r"(?P<day>\d{1,2})[-/.](?P<month>\d{1,2})[-/.](?P<year>(?:19|20)\d{2})",
            value,
        )
        if day_month_year:
            return datetime(
                int(day_month_year.group("year")),
                int(day_month_year.group("month")),
                int(day_month_year.group("day")),
            )

        month_year = re.fullmatch(r"(?P<month>\d{1,2})[-/.](?P<year>(?:19|20)\d{2})", value)
        if month_year:
            return datetime(int(month_year.group("year")), int(month_year.group("month")), 1)

        year_only = re.fullmatch(r"(?:19|20)\d{2}", value)
        if year_only:
            return datetime(int(value), 1, 1)
    except Exception:
        return None

    for date_format in MONTH_FORMATS:
        try:
            return datetime.strptime(value, date_format)
        except Exception:
            continue

    return None


def months_between(start: str, end: str) -> Optional[int]:
    start_date = _parse_year_month(start) if start else None
    end_date = _parse_year_month(end) if end else None
    if not start_date or not end_date:
        return None

    return max(0, (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month))


MONTH_NAME = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z.]*"
YEAR = r"(?:19|20)\d{2}"
NUMERIC_DATE = rf"(?:{YEAR}[-/.]\d{{1,2}}(?:[-/.]\d{{1,2}})?|\d{{1,2}}[-/.]\d{{1,2}}[-/.]{YEAR}|\d{{1,2}}[-/.]{YEAR}|{YEAR})"
TEXT_DATE = rf"(?:{MONTH_NAME}\s+\d{{1,2}},?\s+{YEAR}|\d{{1,2}}\s+{MONTH_NAME},?\s+{YEAR}|{MONTH_NAME}\s+{YEAR}|{YEAR}\s+{MONTH_NAME})"
DATE_TOKEN = rf"(?:to date|hien tai|present|current|now|nay|{TEXT_DATE}|{NUMERIC_DATE})"
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN})\s*(?:-|to|until)\s*(?P<end>{DATE_TOKEN})",
    re.IGNORECASE,
)
LOOSE_PRESENT_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{TEXT_DATE}|{NUMERIC_DATE})\s+(?P<end>to date|hien tai|present|current|now|nay)\b",
    re.IGNORECASE,
)


def normalize_date_label(value: str) -> str | None:
    cleaned = _clean_date_value(value)
    if cleaned in PRESENT_VALUES:
        return "present"

    parsed = _parse_year_month(value)
    if not parsed:
        return None

    return f"{parsed.year:04d}-{parsed.month:02d}"


def parse_explicit_duration_months(text: str) -> int | None:
    normalized = strip_accents(_normalize_date_text(text or "")).lower()

    year_match = re.search(r"(?P<years>\d+(?:\.\d+)?)\s*(?:years?|yrs?|nam)", normalized)
    month_match = re.search(r"(?P<months>\d+)\s*(?:months?|mos?|thang)", normalized)

    if not year_match and not month_match:
        return None

    total = 0
    if year_match:
        total += round(float(year_match.group("years")) * 12)
    if month_match:
        total += int(month_match.group("months"))

    return total


def extract_date_range(text: str) -> dict[str, str | int | None] | None:
    normalized = strip_accents(_normalize_date_text(text or ""))
    match = DATE_RANGE_RE.search(normalized) or LOOSE_PRESENT_DATE_RANGE_RE.search(normalized)
    if not match:
        return None

    start_date = normalize_date_label(match.group("start"))
    end_date = normalize_date_label(match.group("end"))

    return {
        "raw": match.group(0),
        "start_date": start_date,
        "end_date": end_date,
        "duration_months": months_between(start_date, end_date) if start_date and end_date else None,
    }
