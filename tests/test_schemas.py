import pytest
from pydantic import ValidationError

from app.schemas.evaluation import EvaluationResult
from app.schemas.job_description import ParsedJobDescriptionData
from app.schemas.resume import (
    ParsedResumeData,
    ResumeAchievement,
    ResumeCertification,
    ResumeEducation,
    ResumeExperience,
    ResumeLanguage,
    ResumeProject,
    ResumeSkill,
)


def test_parsed_resume_data_can_be_created_with_defaults():
    data = ParsedResumeData()

    assert data.personal is not None
    assert data.summary is None
    assert data.skills == []
    assert data.education == []
    assert data.experience == []
    assert data.projects == []
    assert data.certifications == []
    assert data.achievements == []
    assert data.languages == []


def test_resume_skill_requires_name_and_normalized_name():
    skill = ResumeSkill(
        name="FastAPI",
        normalized_name="fastapi",
        category="backend",
        evidence="Mentioned in resume text",
    )

    assert skill.name == "FastAPI"
    assert skill.normalized_name == "fastapi"
    assert skill.category == "backend"
    assert skill.evidence == "Mentioned in resume text"


def test_resume_skill_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        ResumeSkill()


def test_parsed_resume_data_can_be_created_with_full_shape():
    data = ParsedResumeData(
        summary="Backend developer with FastAPI experience.",
        skills=[
            ResumeSkill(
                name="Python",
                normalized_name="python",
                category="backend",
                evidence="Mentioned in resume text",
            )
        ],
        education=[
            ResumeEducation(
                institution="University of Technology",
                degree="Bachelor",
                field_of_study="Computer Science",
                start_year=2019,
                end_year=2023,
                description="Studied software engineering.",
            )
        ],
        experience=[
            ResumeExperience(
                company="ABC Tech",
                role="Backend Developer",
                location="Ho Chi Minh City",
                start_date="2023-01",
                end_date="2024-12",
                duration_months=24,
                responsibilities=[
                    "Developed backend APIs.",
                    "Integrated PostgreSQL and Redis.",
                ],
                technologies=[
                    "Python",
                    "FastAPI",
                    "PostgreSQL",
                ],
            )
        ],
        projects=[
            ResumeProject(
                name="Recruitment Management API",
                description="Built backend APIs for candidate management.",
                technologies=[
                    "Python",
                    "FastAPI",
                ],
                url="https://github.com/example/recruitment-api",
            )
        ],
        certifications=[
            ResumeCertification(
                name="Docker Foundations",
                issuer="Docker",
                issued_year=2024,
                url=None,
            )
        ],
        achievements=[
            ResumeAchievement(
                title="Improved API response time",
                description="Optimized database queries.",
                year=2024,
            )
        ],
        languages=[
            ResumeLanguage(
                name="English",
                proficiency="intermediate",
            )
        ],
    )

    assert data.summary == "Backend developer with FastAPI experience."
    assert data.skills[0].normalized_name == "python"
    assert data.education[0].institution == "University of Technology"
    assert data.experience[0].role == "Backend Developer"
    assert data.projects[0].technologies == ["Python", "FastAPI"]
    assert data.certifications[0].issued_year == 2024
    assert data.achievements[0].title == "Improved API response time"
    assert data.languages[0].name == "English"


def test_parsed_job_description_data_can_be_created_with_defaults():
    data = ParsedJobDescriptionData()

    assert data.responsibilities == []
    assert data.required_skills == []
    assert data.preferred_skills == []
    assert data.domain_keywords == []


def test_evaluation_result_can_be_created():
    result = EvaluationResult(
        overall_score=75.5,
        summary="Candidate is a potential fit.",
    )

    assert result.overall_score == 75.5
    assert result.criteria == []
    assert result.skills == []
    assert result.interview_questions == []
    assert result.evidence_map == {}