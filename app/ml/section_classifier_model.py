from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.ml.section_classifier_features import (
    LABELS,
    align_proba_to_labels,
    build_feature_matrix,
    viterbi_decode,
)

logger = logging.getLogger(__name__)


def _load_sentence_transformer_preferring_cache(base_model: str) -> Any:
    """Load a bare HuggingFace Hub model id, trying the LOCAL cache first.

    `SentenceTransformer(base_model)` by default makes a network HEAD request
    to check for updates even when the model is already fully cached — if
    the Hub is slow/unreachable, huggingface_hub retries with exponential
    backoff across several files before finally falling back to cache,
    which can add minutes of latency to what should be an instant load.
    Observed in practice as this exact model hanging on repeated
    `ReadTimeoutError` retries against huggingface.co.

    Since `base_model` here is a stable, already-downloaded dependency (used
    throughout this project's training scripts), try `local_files_only=True`
    first — instant if cached, raises immediately if not — and only fall
    back to a normal (network-allowed) load for a genuinely fresh
    environment that has never downloaded this model before.
    """
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(base_model, local_files_only=True)
    except Exception:
        logger.info(
            "'%s' not found in local cache — falling back to a network-allowed load "
            "(this may be slow if the Hub is unreachable).",
            base_model,
        )
        return SentenceTransformer(base_model)


def _is_valid_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "classifier.joblib").exists()
        and (path / "transition_matrix.npy").exists()
    )


@dataclass
class SectionClassifierBundle:
    classifier: Any
    transition_log_probs: np.ndarray
    embedder: Any

    def predict_labels(self, lines: list[str]) -> list[str]:
        """Predict a Viterbi-smoothed section label for each line in `lines`
        — `lines` must be a single resume's lines, in order (relative
        position within the sequence is one of the structural features, and
        Viterbi smoothing only makes sense within one document)."""
        if not lines:
            return []

        embeddings = np.asarray(
            self.embedder.encode(lines, normalize_embeddings=True, show_progress_bar=False)
        )
        features = build_feature_matrix(embeddings, lines)
        proba = align_proba_to_labels(
            self.classifier.predict_proba(features), self.classifier.classes_, LABELS
        )
        emission_log_probs = np.log(np.clip(proba, 1e-12, None))
        indices = viterbi_decode(emission_log_probs, self.transition_log_probs)
        return [LABELS[i] for i in indices]


class SectionClassifierLoader:
    def __init__(self, model_path: str, base_model: str) -> None:
        self._model_path = model_path
        self._base_model = base_model

    def load(self) -> SectionClassifierBundle:
        candidate = Path(self._model_path) if self._model_path else None
        if candidate is None or not _is_valid_model_dir(candidate):
            raise RuntimeError(
                f"Section classifier model not found at '{self._model_path}' "
                "(expected a directory containing classifier.joblib and transition_matrix.npy)."
            )

        logger.info("Loading section classifier model from %s", candidate)
        classifier = joblib.load(candidate / "classifier.joblib")
        transition_log_probs = np.load(candidate / "transition_matrix.npy")
        embedder = self._load_embedder()
        return SectionClassifierBundle(
            classifier=classifier,
            transition_log_probs=transition_log_probs,
            embedder=embedder,
        )

    def _load_embedder(self) -> Any:
        """Reuse the similarity-model singleton ONLY when it's guaranteed to
        be the same untouched base embedding model this classifier was
        trained against (enabled, no fine-tuned override, identical base
        model string) — otherwise instantiate an independent
        SentenceTransformer. Sharing with a fine-tuned CV-JD similarity model
        would silently feed this classifier embeddings from a different
        space than it was trained on."""
        try:
            from app.ml.similarity_config import get_similarity_config
            from app.ml.similarity_model import get_similarity_model

            similarity_cfg = get_similarity_config()
            if (
                similarity_cfg.fallback_mode != "rule_only"
                and not similarity_cfg.model_path
                and similarity_cfg.base_model == self._base_model
            ):
                return get_similarity_model()
        except Exception:  # pragma: no cover - defensive: fall through to an independent model
            pass

        return _load_sentence_transformer_preferring_cache(self._base_model)


@lru_cache(maxsize=1)
def get_section_classifier_model() -> SectionClassifierBundle:
    from app.ml.section_classifier_config import get_section_classifier_config

    cfg = get_section_classifier_config()
    if cfg.fallback_mode == "regex_only":
        raise RuntimeError("Section classifier disabled: fallback_mode=regex_only")
    return SectionClassifierLoader(model_path=cfg.model_path, base_model=cfg.base_model).load()
