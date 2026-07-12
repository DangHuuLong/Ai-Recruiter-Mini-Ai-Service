from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.scorers.similarity_scorer import score_cv_jd_similarity


def _make_request():
    from app.schemas.evaluation import EvaluationConfigInput, ScoreApplicationRequest
    from app.schemas.job_description import ParsedJobDescriptionData
    from app.schemas.resume import ParsedResumeData

    return ScoreApplicationRequest(
        resume=ParsedResumeData(),
        job_description=ParsedJobDescriptionData(),
        config=EvaluationConfigInput(criteria=[]),
    )


class TestScoreCvJdSimilarity:
    _PATCH_MODEL = "app.ml.similarity_model.get_similarity_model"

    def test_calls_predict_with_cv_jd_pair_and_scales_to_100(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.75]

        with patch(self._PATCH_MODEL, return_value=mock_model):
            score = score_cv_jd_similarity(_make_request())

        assert score == 75.0
        (call_arg,), _ = mock_model.predict.call_args
        assert len(call_arg) == 1
        cv_text, jd_text = call_arg[0]
        assert isinstance(cv_text, str)
        assert isinstance(jd_text, str)

    def test_clips_negative_raw_score_to_zero(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [-0.2]

        with patch(self._PATCH_MODEL, return_value=mock_model):
            score = score_cv_jd_similarity(_make_request())

        assert score == 0.0

    def test_clips_raw_score_above_one_to_hundred(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.4]

        with patch(self._PATCH_MODEL, return_value=mock_model):
            score = score_cv_jd_similarity(_make_request())

        assert score == 100.0
