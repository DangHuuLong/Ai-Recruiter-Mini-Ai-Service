import re

from app.parsers.normalizer import normalize_text, split_lines, strip_accents
from app.parsers.extractors.achievements import extract_achievements
from app.parsers.extractors.education import extract_education
from app.parsers.extractors.email_phone import extract_emails, extract_phones
from app.parsers.extractors.experience import extract_experience
from app.parsers.extractors.languages import extract_languages
from app.parsers.extractors.links import extract_links
from app.parsers.extractors.projects_certifications import extract_certifications, extract_projects
from app.parsers.extractors.skills import extract_skills
from app.parsers.normalizer import normalize_text, split_lines
from app.parsers.normalizers.skill_normalizer import normalize_skill
from app.parsers.section_splitter import split_sections
from app.schemas.resume import (
    ParsedResumeData,
    ResumeAchievement,
    ResumeCertification,
    ResumeEducation,
    ResumeExperience,
    ResumeLanguage,
    ResumePersonalInfo,
    ResumeProject,
    ResumeSkill,
)

LOCATION_LABELS = (
    "location",
    "address",
    "current address",
    "dia chi",
    "địa chỉ",
    "que quan",
    "quê quán",
    "noi o",
    "nơi ở",
    "thanh pho",
    "thành phố",
)


def _extract_location(raw_text: str) -> str | None:
    for raw_line in split_lines(raw_text):
        line = raw_line.strip()
        normalized = strip_accents(line).lower()

        if ":" in line:
            label, value = line.split(":", 1)
            normalized_label = strip_accents(label).lower().strip()

            if any(token in normalized_label for token in LOCATION_LABELS):
                cleaned = value.strip(" -|,")
                return cleaned or None

        if any(label in normalized for label in ["da nang", "đà nẵng", "ha noi", "hà nội", "ho chi minh", "hcm"]):
            if len(line.split()) <= 10 and not any(token in normalized for token in ["email", "@", "sdt", "phone"]):
                return line

    return None


def _extract_full_name(raw_text: str) -> str | None:
    for line in split_lines(raw_text):
        lowered = line.lower()
        if any(token in lowered for token in ["@", "http", "linkedin", "github", "phone", "email"]):
            continue
        if any(token in lowered for token in ["developer", "engineer", "manager", "designer", "analyst", "consultant"]):
            continue
        if any(char.isdigit() for char in line):
            continue
        if ":" in line:
            continue

        words = line.split()
        if 2 <= len(words) <= 5:
            return line

    return None


def _section_or_fallback(sections: dict[str, str], section_name: str) -> str:
    return sections.get(section_name) or sections.get("other", "")


def parse_resume(raw_text: str) -> ParsedResumeData:
    """Parse resume free text into ParsedResumeData using deterministic heuristics."""
    normalized_text = normalize_text(raw_text, lowercase=False, remove_accents=False, preserve_lines=True)
    sections = split_sections(normalized_text)

    personal = ResumePersonalInfo()
    emails = extract_emails(raw_text)
    phones = extract_phones(raw_text)
    links = extract_links(raw_text)

    if emails:
        personal.email = emails[0]
    if phones:
        personal.phone = phones[0]
    personal.full_name = _extract_full_name(raw_text)
    personal.location = _extract_location(raw_text)
    personal.linkedin_url = links.get("linkedin")
    personal.github_url = links.get("github")
    personal.portfolio_url = links.get("portfolio")

    skills = [
        ResumeSkill(
            name=skill["name"],
            normalized_name=normalize_skill(skill["name"]),
            category=skill.get("category"),
            evidence=skill.get("evidence"),
        )
        for skill in extract_skills(raw_text)
        if skill.get("name")
    ]

    education = [
        ResumeEducation(
            institution=item.get("institution"),
            degree=item.get("degree"),
            field_of_study=item.get("field_of_study"),
            start_year=item.get("start_year"),
            end_year=item.get("end_year"),
            description=item.get("description"),
        )
        for item in extract_education(_section_or_fallback(sections, "education"))
    ]

    experience = [
        ResumeExperience(
            company=item.get("company"),
            role=item.get("role"),
            start_date=item.get("start_date"),
            end_date=item.get("end_date"),
            duration_months=item.get("duration_months"),
            responsibilities=item.get("responsibilities", []),
            technologies=item.get("technologies", []),
        )
        for item in extract_experience(_experience_source(sections))
    ]

    projects = [
        ResumeProject(
            name=item.get("name"),
            description=item.get("description"),
            technologies=item.get("technologies", []),
            url=item.get("url"),
        )
        for item in extract_projects(sections.get("projects", ""))
    ]

    certifications = [
        ResumeCertification(
            name=item.get("name"),
            issuer=item.get("issuer"),
            issued_year=item.get("issued_year"),
            url=item.get("url"),
        )
        for item in extract_certifications(_section_or_fallback(sections, "certifications"))
    ]

    achievements = [
        ResumeAchievement(title=item.get("title"), description=item.get("description"), year=item.get("year"))
        for item in extract_achievements(sections.get("achievements", ""))
    ]
    achievements.extend(
        ResumeAchievement(title=item.get("title"), description=item.get("description"), year=item.get("year"))
        for item in extract_achievements(sections.get("other", ""), require_keyword=True)
    )

    languages = [
        ResumeLanguage(name=item["name"], proficiency=item.get("proficiency"))
        for item in extract_languages(_section_or_fallback(sections, "languages"))
        if item.get("name")
    ]

    return ParsedResumeData(
        personal=personal,
        summary=sections.get("summary") or None,
        skills=skills,
        education=education,
        experience=experience,
        projects=projects,
        certifications=certifications,
        achievements=achievements,
        languages=languages,
    )


def parse_resume_mock(raw_text: str) -> ParsedResumeData:
    return parse_resume(raw_text)

def _experience_source(sections: dict[str, str]) -> str:
    if sections.get("experience"):
        return sections["experience"]

    other = sections.get("other", "")
    lines = split_lines(other)

    candidate_lines = []
    for line in lines:
        lowered = line.lower()

        if any(token in lowered for token in ["@", "email", "phone", "sđt", "sdt", "ngày sinh", "ngay sinh", "quê quán", "que quan"]):
            continue

        if " at " in lowered:
            candidate_lines.append(line)
            continue

        if re.search(r"(?:19|20)\d{2}|(?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}", line):
            if any(role in lowered for role in ["developer", "engineer", "intern", "manager", "designer", "analyst"]):
                candidate_lines.append(line)
                continue

        # Nếu đã có current experience, cho phép gom dòng mô tả ngay sau nó
        if candidate_lines and not any(token in lowered for token in ["university", "đại học", "dai hoc", "certified", "certificate"]):
            candidate_lines.append(line)

    return "\n".join(candidate_lines)