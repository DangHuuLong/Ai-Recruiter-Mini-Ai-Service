from app.schemas.resume import (
    ParsedResumeData,
    ResumePersonalInfo,
    ResumeSkill,
    ResumeExperience,
    ResumeProject,
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
        experience=[
            ResumeExperience(
                company="Mock Company",
                title="Backend Developer",
                start_date=None,
                end_date=None,
                description="Mock backend development experience extracted from resume text.",
                skills=[skill.name for skill in skills],
            )
        ],
        projects=[
            ResumeProject(
                name="Mock Backend API Project",
                description="Mock project extracted from resume text.",
                technologies=[skill.name for skill in skills],
                role="Backend Developer",
            )
        ],
        education=[],
        certifications=[],
        achievements=[],
        languages=[],
    )