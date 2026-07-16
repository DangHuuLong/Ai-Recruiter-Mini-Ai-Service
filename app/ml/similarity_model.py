from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
except ImportError:  # pragma: no cover
    _CrossEncoder = None  # type: ignore[assignment, misc]


def _is_valid_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def _load_cross_encoder_preferring_cache(base_model: str) -> Any:
    """Load a bare HuggingFace Hub CrossEncoder id, trying the local cache
    first — see app.ml.section_classifier_model._load_sentence_transformer_preferring_cache
    for why: a Hub HEAD-request freshness check can hang for minutes under
    network issues even when the model is already fully cached locally."""
    try:
        return _CrossEncoder(base_model, local_files_only=True)
    except Exception:
        logger.info(
            "'%s' not found in local cache — falling back to a network-allowed load "
            "(this may be slow if the Hub is unreachable).",
            base_model,
        )
        return _CrossEncoder(base_model)


class SimilarityModelLoader:
    """Loads the CV-JD relevance model as a `sentence_transformers.CrossEncoder`
    (pointwise pair scorer — trained via training/fine_tune_cross_encoder_*
    notebooks, see docs/similarity-model-cross-encoder-v0.6.md), not a
    bi-encoder. `model_path` must point at a CrossEncoder-format checkpoint
    (a HuggingFace `*ForSequenceClassification` dir), and `base_model` must be
    a CrossEncoder-compatible base (e.g. `cross-encoder/ms-marco-MiniLM-L-12-v2`)
    — a plain sentence-embedding bi-encoder checkpoint/name is NOT compatible
    here despite loading without error, since it lacks a classification head."""

    def __init__(self, model_path: str, base_model: str) -> None:
        self._model_path = model_path
        self._base_model = base_model

    def load(self) -> Any:
        if _CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

        candidate = Path(self._model_path) if self._model_path else None
        if candidate is not None and _is_valid_model_dir(candidate):
            logger.info("Loading fine-tuned CrossEncoder similarity model from %s", candidate)
            return _CrossEncoder(str(candidate))

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

        return _load_cross_encoder_preferring_cache(self._base_model)


@lru_cache(maxsize=1)
def get_similarity_model() -> Any:
    from app.ml.similarity_config import get_similarity_config

    cfg = get_similarity_config()
    if cfg.fallback_mode == "rule_only":
        raise RuntimeError("Similarity model disabled: fallback_mode=rule_only")
    return SimilarityModelLoader(
        model_path=cfg.model_path,
        base_model=cfg.base_model,
    ).load()
