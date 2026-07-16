from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ml.section_classifier_features import LABELS
from app.ml.section_classifier_model import (
    SectionClassifierBundle,
    SectionClassifierLoader,
    _is_valid_model_dir,
    get_section_classifier_model,
)


class TestIsValidModelDir:
    def test_missing_directory(self, tmp_path: Path) -> None:
        assert not _is_valid_model_dir(tmp_path / "nonexistent")

    def test_missing_artifacts(self, tmp_path: Path) -> None:
        assert not _is_valid_model_dir(tmp_path)

    def test_missing_transition_matrix_only(self, tmp_path: Path) -> None:
        (tmp_path / "classifier.joblib").write_bytes(b"")
        assert not _is_valid_model_dir(tmp_path)

    def test_valid_model_directory(self, tmp_path: Path) -> None:
        (tmp_path / "classifier.joblib").write_bytes(b"")
        (tmp_path / "transition_matrix.npy").write_bytes(b"")
        assert _is_valid_model_dir(tmp_path)


class TestSectionClassifierLoader:
    def test_raises_when_path_empty(self) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            SectionClassifierLoader(model_path="", base_model="base-model").load()

    def test_raises_when_path_missing_artifacts(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            SectionClassifierLoader(model_path=str(tmp_path), base_model="base-model").load()

    def test_loads_artifacts_when_valid(self, tmp_path: Path) -> None:
        (tmp_path / "classifier.joblib").write_bytes(b"")
        np.save(tmp_path / "transition_matrix.npy", np.zeros((len(LABELS), len(LABELS))))

        mock_classifier = MagicMock()
        mock_embedder = MagicMock()

        with patch(
            "app.ml.section_classifier_model.joblib.load", return_value=mock_classifier
        ) as mock_joblib_load, patch.object(
            SectionClassifierLoader, "_load_embedder", return_value=mock_embedder
        ):
            bundle = SectionClassifierLoader(model_path=str(tmp_path), base_model="base-model").load()

        mock_joblib_load.assert_called_once_with(tmp_path / "classifier.joblib")
        assert bundle.classifier is mock_classifier
        assert bundle.embedder is mock_embedder
        assert bundle.transition_log_probs.shape == (len(LABELS), len(LABELS))


class TestSectionClassifierLoaderEmbedderSharing:
    """_load_embedder must only reuse the similarity-model singleton when
    it's guaranteed to be the same untouched base embedding model — never
    when a fine-tuned override or a different base model is in play."""

    def test_shares_similarity_singleton_when_compatible(self) -> None:
        mock_similarity_cfg = MagicMock(fallback_mode="base_model", model_path="", base_model="shared-model")
        mock_shared_model = MagicMock()

        with patch("app.ml.similarity_config.get_similarity_config", return_value=mock_similarity_cfg), \
             patch("app.ml.similarity_model.get_similarity_model", return_value=mock_shared_model):
            embedder = SectionClassifierLoader(model_path="x", base_model="shared-model")._load_embedder()

        assert embedder is mock_shared_model

    def test_does_not_share_when_similarity_uses_finetuned_path(self) -> None:
        mock_similarity_cfg = MagicMock(fallback_mode="base_model", model_path="/some/finetuned", base_model="shared-model")

        with patch("app.ml.similarity_config.get_similarity_config", return_value=mock_similarity_cfg), \
             patch("sentence_transformers.SentenceTransformer") as mock_st:
            SectionClassifierLoader(model_path="x", base_model="shared-model")._load_embedder()

        mock_st.assert_called_once_with("shared-model", local_files_only=True)

    def test_does_not_share_when_similarity_disabled(self) -> None:
        mock_similarity_cfg = MagicMock(fallback_mode="rule_only", model_path="", base_model="shared-model")

        with patch("app.ml.similarity_config.get_similarity_config", return_value=mock_similarity_cfg), \
             patch("sentence_transformers.SentenceTransformer") as mock_st:
            SectionClassifierLoader(model_path="x", base_model="shared-model")._load_embedder()

        mock_st.assert_called_once_with("shared-model", local_files_only=True)

    def test_does_not_share_when_base_model_differs(self) -> None:
        mock_similarity_cfg = MagicMock(fallback_mode="base_model", model_path="", base_model="other-model")

        with patch("app.ml.similarity_config.get_similarity_config", return_value=mock_similarity_cfg), \
             patch("sentence_transformers.SentenceTransformer") as mock_st:
            SectionClassifierLoader(model_path="x", base_model="shared-model")._load_embedder()

        mock_st.assert_called_once_with("shared-model", local_files_only=True)


class TestSectionClassifierBundlePredictLabels:
    def test_empty_lines_returns_empty(self) -> None:
        bundle = SectionClassifierBundle(
            classifier=MagicMock(),
            transition_log_probs=np.zeros((len(LABELS), len(LABELS))),
            embedder=MagicMock(),
        )
        assert bundle.predict_labels([]) == []

    def test_predicts_and_smooths(self) -> None:
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.zeros((3, 4))

        mock_classifier = MagicMock()
        mock_classifier.classes_ = np.array(list(LABELS))
        proba = np.zeros((3, len(LABELS)))
        proba[:, LABELS.index("skills")] = 1.0
        mock_classifier.predict_proba.return_value = proba

        transition_log_probs = np.log(np.full((len(LABELS), len(LABELS)), 1 / len(LABELS)))

        bundle = SectionClassifierBundle(
            classifier=mock_classifier,
            transition_log_probs=transition_log_probs,
            embedder=mock_embedder,
        )
        labels = bundle.predict_labels(["a", "b", "c"])

        assert labels == ["skills", "skills", "skills"]


class TestGetSectionClassifierModel:
    def setup_method(self) -> None:
        # Guard against cache pollution from OTHER test modules that exercise
        # the real model via the actual .env config (e.g. tests that call
        # parse_resume() -> split_sections() with SECTION_CLASSIFIER_* enabled)
        # — teardown_method alone only cleans up after THIS class's own
        # tests, not state inherited from whatever ran before it.
        get_section_classifier_model.cache_clear()

    def teardown_method(self) -> None:
        get_section_classifier_model.cache_clear()

    def test_raises_when_disabled(self) -> None:
        mock_cfg = MagicMock(fallback_mode="regex_only")
        with patch("app.ml.section_classifier_config.get_section_classifier_config", return_value=mock_cfg):
            with pytest.raises(RuntimeError, match="regex_only"):
                get_section_classifier_model()

    def test_loads_and_caches_when_enabled(self, tmp_path: Path) -> None:
        mock_cfg = MagicMock(
            fallback_mode="model_with_regex_fallback",
            model_path=str(tmp_path),
            base_model="base-model",
        )
        mock_bundle = MagicMock()

        with patch(
            "app.ml.section_classifier_config.get_section_classifier_config", return_value=mock_cfg
        ), patch.object(SectionClassifierLoader, "load", return_value=mock_bundle) as mock_load:
            result1 = get_section_classifier_model()
            result2 = get_section_classifier_model()

        assert result1 is mock_bundle
        assert result2 is mock_bundle
        mock_load.assert_called_once()
