from app.schemas.resume import (
    ParsedResumeData,
    ResumeExperience,
    ResumePersonalInfo,
    ResumeProject,
    ResumeSkill,
    ResumeEducation,
    ResumeCertification,
    ResumeAchievement,
    ResumeLanguage,
)

from app.parsers.normalizer import normalize_text, split_lines
from app.parsers.section_splitter import split_sections
from app.parsers.extractors.email_phone import extract_emails, extract_phones
from app.parsers.extractors.links import extract_links
from app.parsers.extractors.skills import extract_skills


def parse_resume(raw_text: str) -> ParsedResumeData:
    """Orchestrate lightweight parsing of resume free text into ParsedResumeData.

    This implementation is intentionally simple and rule-based: it normalizes text,
    splits into sections and runs small extractors to populate the schema.
    """
    normalized = normalize_text(raw_text, lowercase=False, remove_accents=False)
    sections = split_sections(normalized)

    # Personal
    personal = ResumePersonalInfo()
    # try to extract emails/phones/links from whole text
    emails = extract_emails(raw_text)
    phones = extract_phones(raw_text)
    links = extract_links(raw_text)

    if emails:
        personal.email = emails[0]
    if phones:
        personal.phone = phones[0]
    personal.linkedin_url = links.get("linkedin")
    personal.github_url = links.get("github")
    personal.portfolio_url = links.get("portfolio")

    # Summary
    summary = sections.get("summary") or None

    # Skills
    skill_candidates = []
    # prefer skills section if available
    skills_section = sections.get("skills") or raw_text
    raw_skills = extract_skills(skills_section)
    for s in raw_skills:
        skill_candidates.append(
            ResumeSkill(name=s.get("name"), normalized_name=s.get("normalized_name"), evidence=s.get("evidence"))
        )

    # Education/Experience/Projects minimal parsing: keep lines as entries
    education = []
    if sections.get("education"):
        for line in split_lines(sections.get("education")):
            education.append(ResumeEducation(institution=line))

    experience = []
    if sections.get("experience"):
        for line in split_lines(sections.get("experience")):
            experience.append(ResumeExperience(company=None, role=line, responsibilities=[]))

    projects = []
    if sections.get("projects"):
        for line in split_lines(sections.get("projects")):
            projects.append(ResumeProject(name=line))

    certifications = []
    if sections.get("certifications"):
        for line in split_lines(sections.get("certifications")):
            certifications.append(ResumeCertification(name=line))

    achievements = []
    if sections.get("other"):
        # opportunistic: look for lines that include 'award' or 'achieve'
        for line in split_lines(sections.get("other")):
            l = line.lower()
            if "award" in l or "achiev" in l or "giải" in l:
                achievements.append(ResumeAchievement(title=line))

    languages = []
    if sections.get("languages"):
        for line in split_lines(sections.get("languages")):
            languages.append(ResumeLanguage(name=line))

    return ParsedResumeData(
        personal=personal,
        summary=summary,
        skills=skill_candidates,
        education=education,
        experience=experience,
        projects=projects,
        certifications=certifications,
        achievements=achievements,
        languages=languages,
    )


def parse_resume_mock(raw_text: str) -> ParsedResumeData:
    # keep mock for fallback/tests
    from app.parsers.resume_parser import parse_resume as _real  # noqa: F401

    # original mock preserved for compatibility
    return ParsedResumeData()