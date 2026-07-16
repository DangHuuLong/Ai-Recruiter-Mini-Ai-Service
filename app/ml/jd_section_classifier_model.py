from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.ml.jd_section_classifier_features import (
    LABELS,
    align_proba_to_labels,
    build_feature_matrix,
    viterbi_decode,
)

logger = logging.getLogger(__name__)


def _is_valid_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "classifier.joblib").exists()
        and (path / "transition_matrix.npy").exists()
    )


@dataclass
class JdSectionClassifierBundle:
    classifier: Any
    transition_log_probs: np.ndarray
    embedder: Any

    def predict_labels(self, lines: list[str]) -> list[str]:
        """Predict a Viterbi-smoothed section label for each line in `lines`
        — `lines` must be a single JD's lines, in order."""
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


class JdSectionClassifierLoader:
    def __init__(self, model_path: str, base_model: str) -> None:
        self._model_path = model_path
        self._base_model = base_model

    def load(self) -> JdSectionClassifierBundle:
        candidate = Path(self._model_path) if self._model_path else None
        if candidate is None or not _is_valid_model_dir(candidate):
            raise RuntimeError(
                f"JD section classifier model not found at '{self._model_path}' "
                "(expected a directory containing classifier.joblib and transition_matrix.npy)."
            )

        logger.info("Loading JD section classifier model from %s", candidate)
        classifier = joblib.load(candidate / "classifier.joblib")
        transition_log_probs = np.load(candidate / "transition_matrix.npy")
        embedder = self._load_embedder()
        return JdSectionClassifierBundle(
            classifier=classifier,
            transition_log_probs=transition_log_probs,
            embedder=embedder,
        )

    def _load_embedder(self) -> Any:
        """Reuse the CV section-classifier's embedder singleton when it's
        enabled and configured with the same base model — avoids loading a
        second identical SentenceTransformer in the same process. Falls
        back to an independent instance otherwise (CV classifier disabled,
        or a different base model configured). Note: the live similarity
        scorer (app/ml/similarity_model.py) is a CrossEncoder, not usable
        here as an embedding model, so it is never a sharing candidate."""
        try:
            from app.ml.section_classifier_config import get_section_classifier_config
            from app.ml.section_classifier_model import get_section_classifier_model

            cv_cfg = get_section_classifier_config()
            if cv_cfg.fallback_mode != "regex_only" and cv_cfg.base_model == self._base_model:
                return get_section_classifier_model().embedder
        except Exception:  # pragma: no cover - defensive: fall through to an independent model
            pass

        from app.ml.section_classifier_model import _load_sentence_transformer_preferring_cache

        return _load_sentence_transformer_preferring_cache(self._base_model)


@lru_cache(maxsize=1)
def get_jd_section_classifier_model() -> JdSectionClassifierBundle:
    from app.ml.jd_section_classifier_config import get_jd_section_classifier_config

    cfg = get_jd_section_classifier_config()
    if cfg.fallback_mode == "regex_only":
        raise RuntimeError("JD section classifier disabled: fallback_mode=regex_only")
    return JdSectionClassifierLoader(model_path=cfg.model_path, base_model=cfg.base_model).load()
