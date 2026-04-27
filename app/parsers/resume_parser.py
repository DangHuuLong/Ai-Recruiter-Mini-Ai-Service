from app.schemas.resume import (
    ParsedResumeData,
    ResumeExperience,
    ResumePersonalInfo,
    ResumeProject,
    ResumeSkill,
)


def parse_resume_mock(raw_text: str) -> ParsedResumeData:
    text_lower = raw_text.lower()

    skills: list[ResumeSkill] = []

    known_skills = [
        ("Python", "python", "backend"),
        ("FastAPI", "fastapi", "backend"),
        ("PostgreSQL", "postgresql", "database"),
        ("Redis", "redis", "cache"),
        ("Docker", "docker", "devops"),
        ("REST API", "rest_api", "backend"),
    ]

    for name, normalized_name, category in known_skills:
        if name.lower() in text_lower:
            skills.append(
                ResumeSkill(
                    name=name,
                    normalized_name=normalized_name,
                    category=category,
                    evidence=f"{name} mentioned in resume text",
                )
            )

    skill_names = [skill.name for skill in skills]

    return ParsedResumeData(
        personal=ResumePersonalInfo(
            full_name="Mock Candidate",
            email=None,
            phone=None,
            location=None,
            linkedin_url=None,
            github_url=None,
            portfolio_url=None,
        ),
        summary="Mock parsed resume profile.",
        skills=skills,
        education=[],
        experience=[
            ResumeExperience(
                company="Mock Company",
                role="Backend Developer",
                location=None,
                start_date=None,
                end_date=None,
                duration_months=None,
                responsibilities=[
                    "Mock backend development experience extracted from resume text."
                ],
                technologies=skill_names,
            )
        ],
        projects=[
            ResumeProject(
                name="Mock Backend API Project",
                description="Mock project extracted from resume text.",
                technologies=skill_names,
                url=None,
            )
        ],
        certifications=[],
        achievements=[],
        languages=[],
    )