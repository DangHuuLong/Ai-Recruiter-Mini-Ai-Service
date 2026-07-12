from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.ml.section_classifier_config import SectionClassifierConfig, get_section_classifier_config


class TestSectionClassifierConfigDefaults:
    def test_default_values(self) -> None:
        cfg = SectionClassifierConfig()
        assert cfg.model_path == ""
        assert cfg.base_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.fallback_mode == "regex_only"

    def test_invalid_fallback_mode_raises(self) -> None:
        with pytest.raises(ValidationError):
            SectionClassifierConfig(fallback_mode="invalid_mode")  # type: ignore[arg-type]


class TestGetSectionClassifierConfig:
    def test_reads_all_fields_from_settings(self) -> None:
        mock_settings = MagicMock()
        mock_settings.section_classifier_model_path = "/models/section-classifier-v0.1"
        mock_settings.section_classifier_base_model = "base-model"
        mock_settings.section_classifier_fallback_mode = "model_with_regex_fallback"

        with patch("app.core.config.get_settings", return_value=mock_settings):
            cfg = get_section_classifier_config()

        assert cfg.model_path == "/models/section-classifier-v0.1"
        assert cfg.base_model == "base-model"
        assert cfg.fallback_mode == "model_with_regex_fallback"

    def test_defaults_to_regex_only(self) -> None:
        mock_settings = MagicMock()
        mock_settings.section_classifier_model_path = ""
        mock_settings.section_classifier_base_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_settings.section_classifier_fallback_mode = "regex_only"

        with patch("app.core.config.get_settings", return_value=mock_settings):
            cfg = get_section_classifier_config()

        assert cfg.fallback_mode == "regex_only"
