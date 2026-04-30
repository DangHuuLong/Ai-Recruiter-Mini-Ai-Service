import re

from app.parsers.extractors.achievements import extract_achievements
from app.parsers.extractors.education import extract_education
from app.parsers.extractors.email_phone import extract_emails, extract_phones
from app.parsers.extractors.experience import extract_experience
from app.parsers.extractors.languages import extract_languages
from app.parsers.extractors.links import extract_links
from app.parsers.extractors.projects_certifications import extract_certifications, extract_projects
from app.parsers.extractors.skills import extract_skills
from app.parsers.normalizer import normalize_text, split_lines, strip_accents
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

EMPTY_LOCATION_VALUES = {
    "",
    "birthday",
    "gender",
    "phone",
    "email",
    "skills",
    "education",
    "hcmus",
}


def _extract_location(raw_text: str) -> str | None:
    lines = split_lines(raw_text)

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        normalized = strip_accents(line).lower()

        if ":" in line:
            label, value = line.split(":", 1)
            normalized_label = strip_accents(label).lower().strip()

            if any(token in normalized_label for token in LOCATION_LABELS):
                cleaned = value.strip(" -|,")
                normalized_value = strip_accents(cleaned).lower().strip()
                return cleaned if normalized_value not in EMPTY_LOCATION_VALUES else None

        normalized_line = strip_accents(line).lower().strip()
        if normalized_line in {"address", "dia chi", "địa chỉ"}:
            next_value = lines[index + 1].strip() if index + 1 < len(lines) else ""
            normalized_next = strip_accents(next_value).lower().strip()
            if normalized_next in EMPTY_LOCATION_VALUES:
                return None
            if any(token in normalized_next for token in ["phone", "email", "birthday", "gender", "skills"]):
                return None
            return next_value or None

        if any(label in normalized for label in ["da nang", "đà nẵng", "ha noi", "hà nội", "ho chi minh"]):
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


def _combine_sections(*values: str | None) -> str:
    return "\n".join(value for value in values if value).strip()


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

    projects_source = _projects_source(sections)
    projects = [
        ResumeProject(
            name=item.get("name"),
            description=item.get("description"),
            technologies=item.get("technologies", []),
            url=item.get("url"),
        )
        for item in extract_projects(projects_source)
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

    achievements_source = _combine_sections(sections.get("achievements"), sections.get("experience"), sections.get("projects"))
    achievements = [
        ResumeAchievement(title=item.get("title"), description=item.get("description"), year=item.get("year"))
        for item in extract_achievements(achievements_source)
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
        experience_text = sections["experience"]
        projects_text = sections.get("projects", "")
        project_lines = split_lines(projects_text)
        if project_lines and not _looks_like_real_project_start(project_lines):
            return _combine_sections(experience_text, projects_text)
        return experience_text

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

        if candidate_lines and not any(token in lowered for token in ["university", "đại học", "dai hoc", "certified", "certificate"]):
            candidate_lines.append(line)

    return "\n".join(candidate_lines)


def _projects_source(sections: dict[str, str]) -> str:
    projects_text = sections.get("projects", "")
    lines = split_lines(projects_text)

    if lines and _looks_like_real_project_start(lines):
        return projects_text

    experience_text = sections.get("experience", "")
    if not experience_text:
        return projects_text

    return _extract_project_tail_from_experience(experience_text)


def _looks_like_real_project_start(lines: list[str]) -> bool:
    if not lines:
        return False

    first = strip_accents(lines[0]).lower()
    second = strip_accents(lines[1]).lower() if len(lines) > 1 else ""

    if any(token in first for token in ["career history", "company", "corporation"]):
        return False
    if any(token in first for token in ["developer", "engineer", "manager"]):
        return False
    if any(token in second for token in ["position", "role", "teamsize", "description", "technologies"]):
        return True
    if re.search(r"(?:19|20)\d{2}[/.-]\d{1,2}\s*(?:-|–|—|to)\s*", lines[0]):
        return True

    return False


def _extract_project_tail_from_experience(experience_text: str) -> str:
    lines = split_lines(experience_text)
    for index, line in enumerate(lines):
        normalized = strip_accents(line).lower()
        if re.search(r"(?:system|platform|app|website|project)\b", normalized) and index + 1 < len(lines):
            next_line = strip_accents(lines[index + 1]).lower()
            if any(token in next_line for token in ["position", "role", "teamsize", "description", "technologies"]):
                return "\n".join(lines[index:])
    return ""