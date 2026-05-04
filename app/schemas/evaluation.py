from pydantic import BaseModel, Field

from app.schemas.job_description import ParsedJobDescriptionData
from app.schemas.resume import ParsedResumeData


class EvaluationCriterionInput(BaseModel):
    criterion: str
    weight: float


class EvaluationConfigInput(BaseModel):
    criteria: list[EvaluationCriterionInput]


class ScoreApplicationRequest(BaseModel):
    resume: ParsedResumeData
    job_description: ParsedJobDescriptionData
    config: EvaluationConfigInput


class EvaluationCriterionScore(BaseModel):
    criterion: str
    weight: float
    score_normalized: float
    reason: str | None = None
    evidence: list[str] = Field(default_factory=list)


class EvaluationSkillResult(BaseModel):
    skill_name: str
    normalized_skill_name: str | None = None
    type: str
    importance: str | None = None
    evidence: str | None = None
    note: str | None = None


class EvaluationInterviewQuestion(BaseModel):
    question: str
    category: str | None = None
    linked_skill: str | None = None
    difficulty: str | None = None
    rationale: str | None = None
    display_order: int


class EvaluationResult(BaseModel):
    overall_score: float
    summary: str | None = None
    criteria: list[EvaluationCriterionScore] = Field(default_factory=list)
    skills: list[EvaluationSkillResult] = Field(default_factory=list)
    explanation: str | None = None
    skill_gap_summary: str | None = None
    interview_questions: list[EvaluationInterviewQuestion] = Field(default_factory=list)
    evidence_map: dict[str, list[str]] = Field(default_factory=dict)