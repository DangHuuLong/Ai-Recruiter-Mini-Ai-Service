from app.schemas.evaluation import EvaluationResult
from app.schemas.job_description import ParsedJobDescriptionData
from app.schemas.resume import ParsedResumeData


def test_parsed_resume_data_can_be_created_with_defaults():
    data = ParsedResumeData()

    assert data.personal is not None
    assert data.skills == []
    assert data.education == []
    assert data.experience == []
    assert data.projects == []


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