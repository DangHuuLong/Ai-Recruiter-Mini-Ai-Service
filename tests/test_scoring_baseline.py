from app.scorers.cv_jd_scorer import score_application_mock
from app.schemas.evaluation import (
    EvaluationConfigInput,
    EvaluationCriterionInput,
    ScoreApplicationRequest,
)
from app.schemas.job_description import JobSkill, ParsedJobDescriptionData
from app.schemas.resume import (
    ParsedResumeData,
    ResumeEducation,
    ResumeExperience,
    ResumeProject,
    ResumeSkill,
)


def test_score_application_baseline_returns_complete_evaluation_result_schema():
    request = ScoreApplicationRequest(
        resume=ParsedResumeData(
            summary="Backend developer with Python, FastAPI, PostgreSQL and Docker experience.",
            skills=[
                ResumeSkill(name="Python", normalized_name="python", category="language"),
                ResumeSkill(name="FastAPI", normalized_name="fastapi", category="backend"),
                ResumeSkill(name="PostgreSQL", normalized_name="postgresql", category="database"),
                ResumeSkill(name="Docker", normalized_name="docker", category="devops"),
            ],
            experience=[
                ResumeExperience(
                    company="Acme",
                    role="Backend Developer",
                    duration_months=30,
                    responsibilities=["Built REST API services and database integrations"],
                    technologies=["Python", "FastAPI", "PostgreSQL", "Docker"],
                )
            ],
            projects=[
                ResumeProject(
                    name="Recruiter API",
                    description="Backend REST API for recruitment workflows using PostgreSQL.",
                    technologies=["Python", "FastAPI", "PostgreSQL"],
                )
            ],
            education=[
                ResumeEducation(
                    institution="University",
                    degree="Bachelor",
                    field_of_study="Computer Science",
                )
            ],
        ),
        job_description=ParsedJobDescriptionData(
            title="Backend Developer",
            responsibilities=["Build REST APIs", "Design database interactions"],
            requirements=["Python", "FastAPI", "PostgreSQL", "Redis"],
            nice_to_have=["Docker", "CI/CD"],
            required_skills=[
                JobSkill(name="Python", normalized_name="python", is_core=True, weight_hint=1.0),
                JobSkill(name="FastAPI", normalized_name="fastapi", is_core=True, weight_hint=1.0),
                JobSkill(name="PostgreSQL", normalized_name="postgresql", is_core=True, weight_hint=1.0),
                JobSkill(name="Redis", normalized_name="redis", is_core=False, weight_hint=0.8),
            ],
            preferred_skills=[
                JobSkill(name="Docker", normalized_name="docker", weight_hint=0.5),
                JobSkill(name="CI/CD", normalized_name="ci/cd", weight_hint=0.5),
            ],
            min_experience_years=2,
            education_requirement="Bachelor Computer Science",
            domain_keywords=["backend", "rest api", "database"],
        ),
        config=EvaluationConfigInput(
            criteria=[
                EvaluationCriterionInput(criterion="SKILLS_MATCH", weight=0.35),
                EvaluationCriterionInput(criterion="EXPERIENCE_RELEVANCE", weight=0.30),
                EvaluationCriterionInput(criterion="PROJECT_RELEVANCE", weight=0.15),
                EvaluationCriterionInput(criterion="EDUCATION_CERTIFICATION", weight=0.10),
                EvaluationCriterionInput(criterion="KEYWORD_DOMAIN_ALIGNMENT", weight=0.10),
            ]
        ),
    )

    result = score_application_mock(request)

    assert 0 <= result.overall_score <= 100
    assert len(result.criteria) == 5
    assert {item.criterion for item in result.criteria} == {
        "SKILLS_MATCH",
        "EXPERIENCE_RELEVANCE",
        "PROJECT_RELEVANCE",
        "EDUCATION_CERTIFICATION",
        "KEYWORD_DOMAIN_ALIGNMENT",
    }
    assert all(0 <= item.score_normalized <= 1 for item in result.criteria)
    assert all(0 <= item.weight <= 1 for item in result.criteria)
    assert result.summary
    assert result.explanation
    assert result.skill_gap_summary
    assert result.interview_questions
    assert result.evidence_map["matched_skills"]
    assert "Redis" in result.evidence_map["missing_skills"]
    assert result.evidence_map["strong_points"]
    assert result.evidence_map["weak_points"]

    skill_types = {skill.type for skill in result.skills}
    assert "MATCHED" in skill_types
    assert "MISSING" in skill_types

    calculated_overall = round(
        sum(item.score_normalized * item.weight * 100 for item in result.criteria),
        2,
    )
    assert result.overall_score == calculated_overall
