import re


EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")


def _unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def extract_emails(text: str) -> list[str]:
    return _unique_in_order([match.group(0) for match in EMAIL_RE.finditer(text or "")])


def extract_phones(text: str) -> list[str]:
    phones = []
    for match in PHONE_RE.finditer(text or ""):
        compact = re.sub(r"[^+0-9]", "", match.group(0))
        digits = re.sub(r"\D", "", compact)
        if len(digits) < 10 or len(digits) > 15:
            continue
        phones.append(compact)

    return _unique_in_order(phones)
