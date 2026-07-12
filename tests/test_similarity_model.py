from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.ml.similarity_model import SimilarityModelLoader, _is_valid_model_dir


class TestIsValidModelDir:
    def test_missing_directory(self, tmp_path: Path) -> None:
        assert not _is_valid_model_dir(tmp_path / "nonexistent")

    def test_directory_without_config(self, tmp_path: Path) -> None:
        assert not _is_valid_model_dir(tmp_path)

    def test_valid_model_directory(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}")
        assert _is_valid_model_dir(tmp_path)


class TestSimilarityModelLoader:
    _PATCH = "app.ml.similarity_model._CrossEncoder"

    def test_loads_base_model_when_path_empty(self) -> None:
        with patch(self._PATCH) as mock_cls:
            SimilarityModelLoader(model_path="", base_model="base-model").load()
        mock_cls.assert_called_once_with("base-model")

    def test_loads_base_model_when_path_missing(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nonexistent")
        with patch(self._PATCH) as mock_cls:
            SimilarityModelLoader(model_path=missing, base_model="base-model").load()
        mock_cls.assert_called_once_with("base-model")

    def test_loads_fine_tuned_when_path_valid(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}")
        with patch(self._PATCH) as mock_cls:
            SimilarityModelLoader(model_path=str(tmp_path), base_model="base-model").load()
        mock_cls.assert_called_once_with(str(tmp_path))

    def test_raises_when_sentence_transformers_missing(self) -> None:
        with patch(self._PATCH, None):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                SimilarityModelLoader(model_path="", base_model="base-model").load()
