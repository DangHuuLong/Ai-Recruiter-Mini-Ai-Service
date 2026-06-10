from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.schemas.evaluation import EvaluationResult, ScoreApplicationRequest


def _make_rule_result(score: float) -> EvaluationResult:
    return EvaluationResult(overall_score=score)


def _make_request() -> ScoreApplicationRequest:
    from app.schemas.job_description import ParsedJobDescriptionData
    from app.schemas.resume import ParsedResumeData
    from app.schemas.evaluation import EvaluationConfigInput

    return ScoreApplicationRequest(
        resume=ParsedResumeData(),
        job_description=ParsedJobDescriptionData(),
        config=EvaluationConfigInput(criteria=[]),
    )


class TestScoringServiceBlend:
    _MOCK_RULE = "app.services.scoring_service.score_application_mock"
    _MOCK_ML = "app.services.scoring_service.ScoringService.score_application"

    def test_weight_zero_returns_rule_score(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(70.0)

        with patch("app.services.scoring_service.score_application_mock", return_value=rule_result), \
             patch("app.services.scoring_service.get_settings") as mock_settings:
            mock_settings.return_value.similarity_scoring_weight = 0.0
            result = ScoringService().score_application(request)

        assert result.overall_score == 70.0
        assert result.similarity_score is None

    def test_weight_one_returns_ml_score(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(70.0)

        with patch("app.services.scoring_service.score_application_mock", return_value=rule_result), \
             patch("app.services.scoring_service.get_settings") as mock_settings, \
             patch("app.scorers.similarity_scorer.score_cv_jd_similarity", return_value=90.0):
            mock_settings.return_value.similarity_scoring_weight = 1.0
            result = ScoringService().score_application(request)

        assert result.overall_score == 90.0
        assert result.similarity_score == 90.0

    def test_weight_half_blends_correctly(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(60.0)

        with patch("app.services.scoring_service.score_application_mock", return_value=rule_result), \
             patch("app.services.scoring_service.get_settings") as mock_settings, \
             patch("app.scorers.similarity_scorer.score_cv_jd_similarity", return_value=80.0):
            mock_settings.return_value.similarity_scoring_weight = 0.5
            result = ScoringService().score_application(request)

        assert result.overall_score == 70.0  # (0.5 × 60) + (0.5 × 80)

    def test_ml_failure_falls_back_to_rule_score(self) -> None:
        from app.services.scoring_service import ScoringService

        request = _make_request()
        rule_result = _make_rule_result(65.0)

        with patch("app.services.scoring_service.score_application_mock", return_value=rule_result), \
             patch("app.services.scoring_service.get_settings") as mock_settings, \
             patch("app.scorers.similarity_scorer.score_cv_jd_similarity", side_effect=RuntimeError("model error")):
            mock_settings.return_value.similarity_scoring_weight = 0.5
            result = ScoringService().score_application(request)

        assert result.overall_score == 65.0
        assert result.similarity_score is None
