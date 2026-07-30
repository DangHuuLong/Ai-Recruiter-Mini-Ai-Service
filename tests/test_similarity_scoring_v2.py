from __future__ import annotations

from unittest.mock import patch

from app.ml.similarity_config import SimilarityConfig
from app.schemas.evaluation import EvaluationResult, ScoreApplicationRequest


def _make_rule_result(score: float) -> EvaluationResult:
    return EvaluationResult(overall_score=score)


def _make_request() -> ScoreApplicationRequest:
    from app.schemas.evaluation import EvaluationConfigInput
    from app.schemas.job_description import ParsedJobDescriptionData
    from app.schemas.resume import ParsedResumeData

    return ScoreApplicationRequest(
        resume=ParsedResumeData(),
        job_description=ParsedJobDescriptionData(),
        config=EvaluationConfigInput(criteria=[]),
    )


class TestScoringServiceBlend:
    _MOCK_RULE = "app.services.scoring_service.score_application_mock"
    _MOCK_CFG = "app.services.scoring_service.get_similarity_config"
    _MOCK_ML = "app.scorers.similarity_scorer.score_cv_jd_similarity"

    def test_weight_zero_returns_rule_score(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(70.0)

        with patch(self._MOCK_RULE, return_value=rule_result), \
             patch(self._MOCK_CFG, return_value=SimilarityConfig(scoring_weight=0.0)):
            result = ScoringService().score_application(request)

        assert result.overall_score == 70.0
        assert result.similarity_score is None

    def test_weight_one_returns_ml_score(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(70.0)

        with patch(self._MOCK_RULE, return_value=rule_result), \
             patch(self._MOCK_CFG, return_value=SimilarityConfig(scoring_weight=1.0)), \
             patch(self._MOCK_ML, return_value=90.0):
            result = ScoringService().score_application(request)

        assert result.overall_score == 90.0
        assert result.similarity_score == 90.0

    def test_weight_half_blends_correctly(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(60.0)

        with patch(self._MOCK_RULE, return_value=rule_result), \
             patch(self._MOCK_CFG, return_value=SimilarityConfig(scoring_weight=0.5)), \
             patch(self._MOCK_ML, return_value=80.0):
            result = ScoringService().score_application(request)

        assert result.overall_score == 70.0  # (0.5 × 60) + (0.5 × 80)

    def test_ml_failure_falls_back_to_rule_score(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(65.0)

        with patch(self._MOCK_RULE, return_value=rule_result), \
             patch(self._MOCK_CFG, return_value=SimilarityConfig(scoring_weight=0.5)), \
             patch(self._MOCK_ML, side_effect=RuntimeError("model error")):
            result = ScoringService().score_application(request)

        assert result.overall_score == 65.0
        assert result.similarity_score is None

    def test_ml_score_below_threshold_uses_rule_score(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(65.0)

        with patch(self._MOCK_RULE, return_value=rule_result), \
             patch(self._MOCK_CFG, return_value=SimilarityConfig(scoring_weight=0.5, score_threshold=50.0)), \
             patch(self._MOCK_ML, return_value=30.0):
            result = ScoringService().score_application(request)

        assert result.overall_score == 65.0
        assert result.similarity_score is None

    def test_fallback_mode_rule_only_skips_ml(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(72.0)

        with patch(self._MOCK_RULE, return_value=rule_result), \
             patch(self._MOCK_CFG, return_value=SimilarityConfig(scoring_weight=1.0, fallback_mode="rule_only")):
            result = ScoringService().score_application(request)

        assert result.overall_score == 72.0
        assert result.similarity_score is None
