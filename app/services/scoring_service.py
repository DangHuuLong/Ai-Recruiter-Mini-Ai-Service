from app.scorers.cv_jd_scorer import score_application_mock
from app.schemas.evaluation import EvaluationResult, ScoreApplicationRequest


class ScoringService:
    def score_application(self, request: ScoreApplicationRequest) -> EvaluationResult:
        return score_application_mock(request)


scoring_service = ScoringService()