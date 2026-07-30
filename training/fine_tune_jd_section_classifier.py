"""
Train the JD section-classifier student model: frozen sentence embeddings +
structural features -> logistic-regression head, plus a Viterbi transition
matrix estimated from training label sequences. Mirrors
training/fine_tune_section_classifier.py (the CV version).

Trains entirely on local CPU — same reasoning as the CV version: only a
single batched embedding call per split, not iterative transformer
fine-tuning.

Usage:
  python -m training.fine_tune_jd_section_classifier
  python -m training.fine_tune_jd_section_classifier \\
      --train datasets/jd_section_splitting/v0.1/classifier/jd_section_classifier_train.jsonl \\
      --validation datasets/jd_section_splitting/v0.1/classifier/jd_section_classifier_validation.jsonl \\
      --output models/jd-section-classifier-v0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.jd_section_classifier_features import (  # noqa: E402
    LABEL_TO_INDEX,
    LABELS,
    align_proba_to_labels,
    compute_features,
)

DEFAULT_TRAIN = Path("datasets/jd_section_splitting/v0.1/classifier/jd_section_classifier_train.jsonl")
DEFAULT_VALIDATION = Path("datasets/jd_section_splitting/v0.1/classifier/jd_section_classifier_validation.jsonl")
DEFAULT_OUTPUT = Path("models/jd-section-classifier-v0.1")
DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TRANSITION_SMOOTHING_ALPHA = 1.0


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def estimate_transition_matrix(rows: list[dict], alpha: float = TRANSITION_SMOOTHING_ALPHA) -> np.ndarray:
    """Bigram label-transition log-probabilities estimated from TRAIN split
    label sequences (grouped per JD, ordered by line_index)."""
    K = len(LABELS)
    counts = np.full((K, K), alpha, dtype=np.float64)

    by_doc: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append((r["line_index"], r["label"]))

    for seq in by_doc.values():
        seq.sort(key=lambda x: x[0])
        labels_seq = [lbl for _, lbl in seq]
        for prev, cur in zip(labels_seq, labels_seq[1:]):
            if prev in LABEL_TO_INDEX and cur in LABEL_TO_INDEX:
                counts[LABEL_TO_INDEX[prev], LABEL_TO_INDEX[cur]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    return np.log(probs)


def train(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
) -> None:
    train_rows = _load_jsonl(train_path)
    validation_rows = _load_jsonl(validation_path)

    if not train_rows:
        raise RuntimeError(f"No training rows found in {train_path}")

    print(f"Loading embedding model: {base_model}")
    model = SentenceTransformer(base_model)

    print(f"Computing features for {len(train_rows)} train rows...")
    X_train, y_train = compute_features(train_rows, model)

    print("Fitting classifier head (LogisticRegression)...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    if validation_rows:
        print(f"Computing features for {len(validation_rows)} validation rows...")
        X_val, y_val = compute_features(validation_rows, model)
        proba_val = clf.predict_proba(X_val)
        proba_val_aligned = align_proba_to_labels(proba_val, clf.classes_, LABELS)
        y_pred = [LABELS[i] for i in np.argmax(proba_val_aligned, axis=1)]

        print("\nPer-line classification report (validation, no Viterbi smoothing):")
        report = classification_report(y_val, y_pred, labels=list(LABELS), zero_division=0)
        print(report)
    else:
        report = "(no validation rows available)"
        print("[!] No validation rows — skipping evaluation report.")

    print("Estimating label-transition matrix from train label sequences...")
    transition_log_probs = estimate_transition_matrix(train_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_dir / "classifier.joblib")
    np.save(output_dir / "transition_matrix.npy", transition_log_probs)

    config = {
        "base_model": base_model,
        "labels": list(LABELS),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "n_train_rows": len(train_rows),
        "n_validation_rows": len(validation_rows),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# JD Section Classifier v0.1\n\n"
        f"Base embedding model: `{base_model}`\n\n"
        f"Trained on {len(train_rows)} lines "
        f"({train_path}); validated on {len(validation_rows)} lines ({validation_path}).\n\n"
        "## Validation report (per-line, no Viterbi smoothing)\n\n"
        f"```\n{report}\n```\n",
        encoding="utf-8",
    )

    print(f"\nSaved artifacts to {output_dir}:")
    print(f"  - classifier.joblib")
    print(f"  - transition_matrix.npy")
    print(f"  - config.json")
    print(f"  - README.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the JD section-classifier student model.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
