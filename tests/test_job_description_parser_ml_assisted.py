from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.parsers.job_description_parser import (
    _ml_assisted_split,
    _regex_split_sections,
    _split_sections,
)


class TestMlAssistedSplitNoAmbiguity:
    def test_skips_model_entirely_when_nothing_is_ambiguous(self) -> None:
        text = "RESPONSIBILITIES\nDesign REST APIs\nREQUIREMENTS\n3+ years Python"

        with patch("app.ml.jd_section_classifier_model.get_jd_section_classifier_model") as mock_get_model:
            result = _ml_assisted_split(text)

        mock_get_model.assert_not_called()
        assert result == _regex_split_sections(text)


class TestMlAssistedSplitReconciliation:
    def test_reclassifies_only_the_other_bucket(self) -> None:
        """A no-header preamble (the job title line) falls into regex's
        'other' bucket. The rest of the document has real headers, so those
        lines must stay exactly as the regex assigned them."""
        text = (
            "Senior Backend Developer\n"
            "REQUIREMENTS\n"
            "3+ years Python and FastAPI experience"
        )

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["other"]

        with patch("app.ml.jd_section_classifier_model.get_jd_section_classifier_model", return_value=mock_bundle):
            result = _ml_assisted_split(text)

        assert result["other"] == ["Senior Backend Developer"]
        assert result["requirements"] == ["3+ years Python and FastAPI experience"]

    def test_lines_the_model_reclassifies_move_into_target_section(self) -> None:
        text = (
            "We are looking for a Senior Backend Developer to join our team.\n"
            "REQUIREMENTS\n"
            "3+ years Python experience"
        )

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["responsibilities"]

        with patch("app.ml.jd_section_classifier_model.get_jd_section_classifier_model", return_value=mock_bundle):
            result = _ml_assisted_split(text)

        assert result["responsibilities"] == ["We are looking for a Senior Backend Developer to join our team."]
        assert result["requirements"] == ["3+ years Python experience"]
        assert result["other"] == []

    def test_reclassified_lines_prepend_existing_section_content(self) -> None:
        text = (
            "Own end-to-end delivery of backend services.\n"
            "RESPONSIBILITIES\n"
            "Design REST APIs"
        )

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["responsibilities"]

        with patch("app.ml.jd_section_classifier_model.get_jd_section_classifier_model", return_value=mock_bundle):
            result = _ml_assisted_split(text)

        assert result["responsibilities"] == [
            "Own end-to-end delivery of backend services.",
            "Design REST APIs",
        ]

    def test_contact_info_lines_protected_from_reclassification(self) -> None:
        text = (
            "Senior Backend Developer\n"
            "Apply at jobs@example.com\n"
            "REQUIREMENTS\n"
            "3+ years Python"
        )

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["responsibilities"]

        with patch("app.ml.jd_section_classifier_model.get_jd_section_classifier_model", return_value=mock_bundle):
            result = _ml_assisted_split(text)

        mock_bundle.predict_labels.assert_called_once_with(["Senior Backend Developer"])
        assert result["other"] == ["Apply at jobs@example.com"]


class TestSplitSectionsFallback:
    def test_falls_back_to_regex_when_model_disabled(self) -> None:
        text = "Senior Backend Developer\nREQUIREMENTS\nPython"

        with patch(
            "app.ml.jd_section_classifier_model.get_jd_section_classifier_model",
            side_effect=RuntimeError("JD section classifier disabled: fallback_mode=regex_only"),
        ):
            result = _split_sections(text)

        assert result == _regex_split_sections(text)

    def test_falls_back_to_regex_on_any_unexpected_error(self) -> None:
        text = "Senior Backend Developer\nREQUIREMENTS\nPython"

        with patch(
            "app.ml.jd_section_classifier_model.get_jd_section_classifier_model",
            side_effect=OSError("model artifact corrupted"),
        ):
            result = _split_sections(text)

        assert result == _regex_split_sections(text)

    def test_uses_ml_result_when_available(self) -> None:
        text = "Senior Backend Developer\nREQUIREMENTS\nPython"

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["other"]

        with patch("app.ml.jd_section_classifier_model.get_jd_section_classifier_model", return_value=mock_bundle):
            result = _split_sections(text)

        assert result["other"] == ["Senior Backend Developer"]
