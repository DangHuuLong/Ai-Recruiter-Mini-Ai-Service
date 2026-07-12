from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.parsers.section_splitter import (
    _ml_assisted_split,
    _regex_split_sections,
    split_sections,
)


class TestMlAssistedSplitNoAmbiguity:
    def test_skips_model_entirely_when_nothing_is_ambiguous(self) -> None:
        """Every line here is claimed by a real header, so the regex baseline
        has an empty 'other' bucket — _ml_assisted_split must return that
        baseline as-is WITHOUT ever calling the classifier model."""
        text = "SKILLS\nPython, Django\nEXPERIENCE\n- Backend Engineer, 2020-2023"

        with patch("app.ml.section_classifier_model.get_section_classifier_model") as mock_get_model:
            result = _ml_assisted_split(text)

        mock_get_model.assert_not_called()
        assert result == _regex_split_sections(text)


class TestMlAssistedSplitReconciliation:
    def test_reclassifies_only_the_other_bucket(self) -> None:
        """A no-header preamble falls into regex's 'other' bucket: its FIRST
        line (name/title convention) stays protected, but the next line is
        genuinely ambiguous content eligible for ML reclassification. The
        rest of the document has real headers, so those lines must stay
        exactly as the regex assigned them."""
        text = (
            "Jane Smith\n"
            "Experienced backend engineer.\n"
            "SKILLS\n"
            "Python, Django, PostgreSQL"
        )

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["summary"]

        with patch("app.ml.section_classifier_model.get_section_classifier_model", return_value=mock_bundle):
            result = _ml_assisted_split(text)

        mock_bundle.predict_labels.assert_called_once_with(["Experienced backend engineer."])
        assert result["summary"] == "Experienced backend engineer."
        assert result["skills"] == "Python, Django, PostgreSQL"
        assert result["other"] == "Jane Smith"

    def test_lines_the_model_still_calls_other_stay_in_other(self) -> None:
        text = "Jane Smith\nSome genuinely unclassifiable separator line\nSKILLS\nPython"

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["other"]

        with patch("app.ml.section_classifier_model.get_section_classifier_model", return_value=mock_bundle):
            result = _ml_assisted_split(text)

        assert result["other"] == "Jane Smith\nSome genuinely unclassifiable separator line"
        assert result["skills"] == "Python"

    def test_lone_first_line_never_invokes_model(self) -> None:
        """A single-line 'other' bucket (just the name/title line, no other
        ambiguous content) has nothing eligible to reclassify — the model
        must not even be called."""
        text = "Jane Smith\nSKILLS\nPython"

        with patch("app.ml.section_classifier_model.get_section_classifier_model") as mock_get_model:
            result = _ml_assisted_split(text)

        mock_get_model.assert_not_called()
        assert result["other"] == "Jane Smith"
        assert result["skills"] == "Python"

    def test_contact_info_lines_protected_even_mid_bucket(self) -> None:
        """An email/phone/URL line anywhere in the 'other' bucket (not just
        the first line) is excluded from reclassification, since the
        training corpus never includes contact info either."""
        text = (
            "Jane Smith\n"
            "Email: jane@example.com\n"
            "Backend engineer with 5 years of experience.\n"
            "SKILLS\n"
            "Python"
        )

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["summary"]

        with patch("app.ml.section_classifier_model.get_section_classifier_model", return_value=mock_bundle):
            result = _ml_assisted_split(text)

        mock_bundle.predict_labels.assert_called_once_with(["Backend engineer with 5 years of experience."])
        assert result["summary"] == "Backend engineer with 5 years of experience."
        assert result["other"] == "Jane Smith\nEmail: jane@example.com"

    def test_reclassified_lines_prepend_existing_section_content(self) -> None:
        """Reclassified 'other' lines occurred earliest in the document, so
        they should end up BEFORE whatever the regex pass already put in the
        target section (e.g. an inline-style second summary line). The
        document's very first line (name) stays protected and is excluded
        from this prepend entirely."""
        text = (
            "Jane Smith\n"
            "Backend engineer with 5 years of experience.\n"
            "Summary: also skilled in DevOps.\n"
            "SKILLS\n"
            "Python"
        )

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["summary"]

        with patch("app.ml.section_classifier_model.get_section_classifier_model", return_value=mock_bundle):
            result = _ml_assisted_split(text)

        assert result["summary"] == "Backend engineer with 5 years of experience.\nalso skilled in DevOps."
        assert result["other"] == "Jane Smith"


class TestSplitSectionsFallback:
    def test_falls_back_to_regex_when_model_disabled(self) -> None:
        text = "Experienced backend engineer.\nSKILLS\nPython"

        with patch(
            "app.ml.section_classifier_model.get_section_classifier_model",
            side_effect=RuntimeError("Section classifier disabled: fallback_mode=regex_only"),
        ):
            result = split_sections(text)

        assert result == _regex_split_sections(text)

    def test_falls_back_to_regex_on_any_unexpected_error(self) -> None:
        text = "Experienced backend engineer.\nSKILLS\nPython"

        with patch(
            "app.ml.section_classifier_model.get_section_classifier_model",
            side_effect=OSError("model artifact corrupted"),
        ):
            result = split_sections(text)

        assert result == _regex_split_sections(text)

    def test_uses_ml_result_when_available(self) -> None:
        text = "Jane Smith\nExperienced backend engineer.\nSKILLS\nPython"

        mock_bundle = MagicMock()
        mock_bundle.predict_labels.return_value = ["summary"]

        with patch("app.ml.section_classifier_model.get_section_classifier_model", return_value=mock_bundle):
            result = split_sections(text)

        assert result["summary"] == "Experienced backend engineer."
        assert result["other"] == "Jane Smith"
