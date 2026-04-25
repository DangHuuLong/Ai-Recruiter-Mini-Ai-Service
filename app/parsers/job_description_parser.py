from app.schemas.job_description import JobSkill, ParsedJobDescriptionData


def parse_job_description_mock(raw_text: str) -> ParsedJobDescriptionData:
    text_lower = raw_text.lower()

    required_skills: list[JobSkill] = []
    preferred_skills: list[JobSkill] = []

    known_required_skills = [
        ("Python", "python"),
        ("FastAPI", "fastapi"),
        ("PostgreSQL", "postgresql"),
        ("REST API", "rest_api"),
    ]

    known_preferred_skills = [
        ("Redis", "redis"),
        ("Docker", "docker"),
    ]

    for name, normalized_name in known_required_skills:
        if name.lower() in text_lower:
            required_skills.append(
                JobSkill(
                    name=name,
                    normalized_name=normalized_name,
                    is_core=True,
                    weight_hint=1.0,
                )
            )

    for name, normalized_name in known_preferred_skills:
        if name.lower() in text_lower:
            preferred_skills.append(
                JobSkill(
                    name=name,
                    normalized_name=normalized_name,
                    is_core=False,
                    weight_hint=0.5,
                )
            )

    return ParsedJobDescriptionData(
        title="Backend Developer",
        seniority="junior",
        employment_type="full-time",
        responsibilities=[
            "Develop and maintain backend APIs",
            "Work with database and service integrations",
        ],
        requirements=[
            "Experience with backend API development",
            "Ability to work with structured data and service layers",
        ],
        nice_to_have=[
            "Experience with Docker or Redis",
        ],
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_experience_years=1,
        education_requirement=None,
        domain_keywords=["backend", "api", "database"],
    )