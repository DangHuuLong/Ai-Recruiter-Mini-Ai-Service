import re


EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")


def extract_emails(text: str) -> list[str]:
    return list({m.group(0) for m in EMAIL_RE.finditer(text)})


def extract_phones(text: str) -> list[str]:
    # simple heuristic: return cleaned matches
    matches = {m.group(0) for m in PHONE_RE.finditer(text)}
    cleaned = [re.sub(r"[^+0-9]", "", m) for m in matches]
    return cleaned
