from __future__ import annotations

import logging

from app.ml.similarity_config import get_similarity_config
from app.scorers.cv_jd_scorer import score_application_mock
from app.schemas.evaluation import EvaluationResult, ScoreApplicationRequest

logger = logging.getLogger(__name__)


class ScoringService:
    def score_application(self, request: ScoreApplicationRequest) -> EvaluationResult:
        rule_result = score_application_mock(request)
        cfg = get_similarity_config()

        if cfg.scoring_weight <= 0.0 or cfg.fallback_mode == "rule_only":
            return rule_result

        try:
            from app.scorers.similarity_scorer import score_cv_jd_similarity

            ml_score = score_cv_jd_similarity(request)
            if ml_score < cfg.score_threshold:
                logger.info(
                    "ML score %.1f below threshold %.1f, skipping blend",
                    ml_score,
                    cfg.score_threshold,
                )
                return rule_result
            blended = round((1.0 - cfg.scoring_weight) * rule_result.overall_score + cfg.scoring_weight * ml_score, 2)
            return rule_result.model_copy(update={"overall_score": blended, "similarity_score": ml_score})
        except Exception as exc:
            logger.warning("ML similarity scoring failed, using rule-based score: %s", exc)
            return rule_result


scoring_service = ScoringService()
