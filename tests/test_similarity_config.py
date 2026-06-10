from __future__ import annotations

import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from app.ml.similarity_config import SimilarityConfig, get_similarity_config


class TestSimilarityConfigDefaults:
    def test_default_values(self) -> None:
        cfg = SimilarityConfig()
        assert cfg.model_path == ""
        assert cfg.base_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.model_version == ""
        assert cfg.scoring_weight == 0.0
        assert cfg.score_threshold == 0.0
        assert cfg.fallback_mode == "base_model"

    def test_scoring_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimilarityConfig(scoring_weight=1.5)

    def test_score_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimilarityConfig(score_threshold=101.0)

    def test_invalid_fallback_mode_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimilarityConfig(fallback_mode="invalid_mode")  # type: ignore[arg-type]


class TestLabelForScore:
    def setup_method(self) -> None:
        self.cfg = SimilarityConfig()

    def test_poor_match(self) -> None:
        assert self.cfg.label_for_score(0.0) == "poor_match"
        assert self.cfg.label_for_score(39.9) == "poor_match"

    def test_weak_match(self) -> None:
        assert self.cfg.label_for_score(40.0) == "weak_match"
        assert self.cfg.label_for_score(59.9) == "weak_match"

    def test_moderate_match(self) -> None:
        assert self.cfg.label_for_score(60.0) == "moderate_match"
        assert self.cfg.label_for_score(74.9) == "moderate_match"

    def test_strong_match(self) -> None:
        assert self.cfg.label_for_score(75.0) == "strong_match"
        assert self.cfg.label_for_score(89.9) == "strong_match"

    def test_excellent_match(self) -> None:
        assert self.cfg.label_for_score(90.0) == "excellent_match"
        assert self.cfg.label_for_score(100.0) == "excellent_match"


class TestGetSimilarityConfig:
    def test_reads_all_fields_from_settings(self) -> None:
        mock_settings = MagicMock()
        mock_settings.similarity_model_path = "/models/v0.2"
        mock_settings.similarity_base_model = "base-model"
        mock_settings.similarity_model_version = "v0.2"
        mock_settings.similarity_scoring_weight = 0.4
        mock_settings.similarity_score_threshold = 55.0
        mock_settings.similarity_fallback_mode = "base_model"

        with patch("app.core.config.get_settings", return_value=mock_settings):
            cfg = get_similarity_config()

        assert cfg.model_path == "/models/v0.2"
        assert cfg.model_version == "v0.2"
        assert cfg.scoring_weight == 0.4
        assert cfg.score_threshold == 55.0
        assert cfg.fallback_mode == "base_model"
