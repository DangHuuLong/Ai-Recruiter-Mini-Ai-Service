"""Shared feature extraction + sequence decoding for the resume section
classifier (student model distilled from an LLM teacher — see
scripts/generate_section_labeling_data.py and
training/fine_tune_section_classifier.py).

This module is imported by BOTH the training script and the runtime model
loader (app/ml/section_classifier_model.py) so the exact same feature
computation is used at train time and serve time — a from-scratch
reimplementation on either side would risk train/serve skew.
"""

from __future__ import annotations

import re

import numpy as np

from app.parsers.normalizer import strip_list_marker
from app.parsers.section_splitter import SECTION_ORDER, matches_known_header

LABELS: tuple[str, ...] = tuple(SECTION_ORDER) + ("other",)
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(LABELS)}

# embedding dims (all-MiniLM-L6-v2) + structural features below
STRUCTURAL_FEATURE_NAMES = (
    "log_word_count",
    "has_bullet_marker",
    "ends_with_colon",
    "caps_ratio",
    "matches_known_header",
    "relative_position",
)

_ALPHA_RE = re.compile(r"[A-Za-z]")


def _has_bullet_marker(line: str) -> bool:
    return strip_list_marker(line) != line.strip()


def _caps_ratio(line: str) -> float:
    letters = _ALPHA_RE.findall(line)
    if not letters:
        return 0.0
    upper = sum(1 for ch in letters if ch.isupper())
    return upper / len(letters)


def structural_features(text: str, index: int, total_lines: int) -> np.ndarray:
    """Fixed-order structural feature vector for one resume line (see
    STRUCTURAL_FEATURE_NAMES for the meaning/order of each entry)."""
    stripped = text.strip()
    word_count = len(stripped.split())
    relative_position = index / max(total_lines - 1, 1)

    return np.array(
        [
            np.log1p(word_count),
            float(_has_bullet_marker(stripped)),
            float(stripped.endswith(":")),
            _caps_ratio(stripped),
            float(matches_known_header(stripped)),
            relative_position,
        ],
        dtype=np.float32,
    )


def build_feature_matrix(embeddings: np.ndarray, texts: list[str]) -> np.ndarray:
    """Concatenate precomputed line embeddings with structural features.

    `texts` must be the flat, resume-grouped, line-index-ordered sequence
    whose position within EACH resume determines `relative_position` — so
    callers must pass one resume's lines at a time (or otherwise ensure
    `total_lines` reflects a single resume's line count), not a
    cross-resume-shuffled batch.
    """
    total = len(texts)
    structural = np.stack(
        [structural_features(t, i, total) for i, t in enumerate(texts)]
    )
    return np.concatenate([embeddings, structural], axis=1)


def compute_features(rows: list[dict], model) -> tuple[np.ndarray, list[str]]:
    """Embed every row's `text` in one batched call, then compute structural
    features per-resume (grouped by `resume_id`, ordered by `line_index` —
    relative_position needs each resume's own line count) and scatter the
    results back into the original row order.

    Shared by training/fine_tune_section_classifier.py and
    training/evaluate_section_classifier.py (and, at inference time,
    app/ml/section_classifier_model.py) so embedding/feature computation
    never diverges between train and serve.

    `model` is any object exposing sentence-transformers' `.encode(...)`
    interface (kept duck-typed here to avoid a hard import dependency in
    this shared module).
    """
    texts = [r["text"] for r in rows]
    embeddings = np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    )

    by_resume: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        by_resume.setdefault(r["resume_id"], []).append(idx)

    feature_dim = embeddings.shape[1] + len(STRUCTURAL_FEATURE_NAMES)
    X = np.zeros((len(rows), feature_dim), dtype=np.float32)

    for resume_id, idxs in by_resume.items():
        idxs = sorted(idxs, key=lambda i: rows[i]["line_index"])
        sub_embeddings = embeddings[idxs]
        sub_texts = [rows[i]["text"] for i in idxs]
        X[idxs] = build_feature_matrix(sub_embeddings, sub_texts)

    labels = [r["label"] for r in rows]
    return X, labels


def align_proba_to_labels(proba: np.ndarray, classes: np.ndarray, labels: tuple[str, ...] = LABELS) -> np.ndarray:
    """Reorder a classifier's predict_proba columns (`classes_` order,
    alphabetical by default) into our canonical LABELS order, filling in a
    near-zero probability for any label absent from the training data
    entirely (a real risk for rare classes early on, before the dataset has
    grown), then renormalize each row to sum to 1."""
    aligned = np.full((proba.shape[0], len(labels)), 1e-9, dtype=np.float64)
    class_to_col = {c: i for i, c in enumerate(classes)}
    for j, label in enumerate(labels):
        if label in class_to_col:
            aligned[:, j] = proba[:, class_to_col[label]]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def viterbi_decode(emission_log_probs: np.ndarray, transition_log_probs: np.ndarray) -> list[int]:
    """Standard Viterbi decode over per-line class log-probabilities.

    emission_log_probs: (T, K) log P(label_t=k | line_t), from the classifier.
    transition_log_probs: (K, K) log P(label_t=k | label_{t-1}=j), indexed
      [j, k] — estimated from training label sequences (see
      training/fine_tune_section_classifier.py's `estimate_transition_matrix`).

    Uses a uniform prior for the first line (no separate start-distribution
    matrix needed — K is small (9), so this stays cheap either way) and
    returns the single most likely label-index sequence of length T.
    """
    T, K = emission_log_probs.shape
    if T == 0:
        return []

    log_delta = np.zeros((T, K), dtype=np.float64)
    backpointer = np.zeros((T, K), dtype=np.int64)
    log_delta[0] = emission_log_probs[0]

    for t in range(1, T):
        scores = log_delta[t - 1][:, None] + transition_log_probs  # (K, K): [prev, cur]
        backpointer[t] = np.argmax(scores, axis=0)
        log_delta[t] = emission_log_probs[t] + np.max(scores, axis=0)

    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(log_delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]

    return path.tolist()
