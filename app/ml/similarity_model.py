from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:  # pragma: no cover
    _SentenceTransformer = None  # type: ignore[assignment, misc]


def _is_valid_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


class SimilarityModelLoader:
    def __init__(self, model_path: str, base_model: str) -> None:
        self._model_path = model_path
        self._base_model = base_model

    def load(self) -> Any:
        if _SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

        candidate = Path(self._model_path) if self._model_path else None
        if candidate is not None and _is_valid_model_dir(candidate):
            logger.info("Loading fine-tuned similarity model from %s", candidate)
            return _SentenceTransformer(str(candidate))

        if self._model_path:
            logger.warning(
                "Fine-tuned model not found at '%s', falling back to base model '%s'",
                self._model_path,
                self._base_model,
            )
        else:
            logger.info(
                "SIMILARITY_MODEL_PATH not set, using base model '%s'",
                self._base_model,
            )

        return _SentenceTransformer(self._base_model)


@lru_cache(maxsize=1)
def get_similarity_model() -> Any:
    from app.core.config import get_settings

    settings = get_settings()
    return SimilarityModelLoader(
        model_path=settings.similarity_model_path,
        base_model=settings.similarity_base_model,
    ).load()
