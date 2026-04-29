"""Heuristic section splitter for resumes.
Finds common headings (Experience, Education, Skills, Projects, Certifications, Languages)
and splits text into a dict of sections.
"""
import re
from typing import Dict, List

HEADERS = {
    "experience": [r"experience", r"work experience", r"professional experience", r"working experience", r"kinh nghiem"],
    "education": [r"education", r"academic", r"hoc van", r"bằng cấp"],
    "skills": [r"skills", r"technical skills", r"ky nang"],
    "projects": [r"projects", r"publications", r"du an"],
    "certifications": [r"certificat", r"certifications", r"chung chi"],
    "languages": [r"languages", r"ngon ngu"],
    "summary": [r"summary", r"profile", r"professional summary", r"tóm tắt"],
}


def find_headings(text: str) -> List[re.Match]:
    pattern = r"^\s*(?P<h>[A-Za-zÀ-ÖØ-öø-ÿ ]{3,50})\s*$"
    return list(re.finditer(pattern, text, flags=re.MULTILINE))


def split_sections(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    sections: Dict[str, List[str]] = {k: [] for k in HEADERS.keys()}
    sections["other"] = []

    current = "other"
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        lowered = s.lower()
        matched = False
        for name, patterns in HEADERS.items():
            for p in patterns:
                if re.search(rf"^{p}$", lowered) or re.search(rf"^{p}:$", lowered):
                    current = name
                    matched = True
                    break
            if matched:
                break

        sections[current].append(raw)

    # join into strings
    return {k: "\n".join(v).strip() for k, v in sections.items()}
