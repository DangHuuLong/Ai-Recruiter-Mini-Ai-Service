from __future__ import annotations

import logging

from app.core.config import get_settings
from app.scorers.cv_jd_scorer import score_application_mock
from app.schemas.evaluation import EvaluationResult, ScoreApplicationRequest

logger = logging.getLogger(__name__)


class ScoringService:
    def score_application(self, request: ScoreApplicationRequest) -> EvaluationResult:
        rule_result = score_application_mock(request)

        settings = get_settings()
        weight = settings.similarity_scoring_weight
        if weight <= 0.0:
            return rule_result

        try:
            from app.scorers.similarity_scorer import score_cv_jd_similarity

            ml_score = score_cv_jd_similarity(request)
            blended = round((1.0 - weight) * rule_result.overall_score + weight * ml_score, 2)
            return rule_result.model_copy(update={
                "overall_score": blended,
                "similarity_score": ml_score,
            })
        except Exception as exc:
            logger.warning("ML similarity scoring failed, using rule-based score: %s", exc)
            return rule_result


scoring_service = ScoringService()
