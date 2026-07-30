"""
Evaluate the trained section-classifier (classifier head + Viterbi
smoothing) on a held-out test split.

Reports per-label precision/recall/F1 (with Viterbi smoothing applied) and a
"boundary accuracy" metric — whether a section CHANGE is correctly detected
at the same line position as the ground truth — which matters more than raw
per-line accuracy here, since section labels are contiguous ordered runs
(see app/ml/section_classifier_features.viterbi_decode) rather than
independent per-line decisions.

Usage:
  python -m training.evaluate_section_classifier
  python -m training.evaluate_section_classifier \\
      --test datasets/section_splitting/v0.1/classifier/section_classifier_test.jsonl \\
      --model models/section-classifier-v0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.section_classifier_features import (  # noqa: E402
    LABELS,
    align_proba_to_labels,
    compute_features,
    viterbi_decode,
)

DEFAULT_TEST = Path("datasets/section_splitting/v0.1/classifier/section_classifier_test.jsonl")
DEFAULT_MODEL_DIR = Path("models/section-classifier-v0.1")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _group_by_resume(rows: list[dict]) -> dict[str, list[dict]]:
    by_resume: dict[str, list[dict]] = {}
    for r in rows:
        by_resume.setdefault(r["resume_id"], []).append(r)
    for resume_id, resume_rows in by_resume.items():
        resume_rows.sort(key=lambda r: r["line_index"])
    return by_resume


def boundary_accuracy(true_labels: list[str], pred_labels: list[str]) -> tuple[int, int]:
    """Correct/total count of whether a section CHANGE is detected at the
    same position as ground truth, across adjacent line pairs. Returns
    (n_correct, n_positions) so callers can aggregate across resumes before
    dividing (avoids skew from resumes with very few lines)."""
    n_correct = 0
    n_positions = len(true_labels) - 1
    for t in range(1, len(true_labels)):
        true_changed = true_labels[t] != true_labels[t - 1]
        pred_changed = pred_labels[t] != pred_labels[t - 1]
        if true_changed == pred_changed:
            n_correct += 1
    return n_correct, n_positions


def evaluate(test_path: Path, model_dir: Path) -> None:
    test_rows = _load_jsonl(test_path)
    if not test_rows:
        raise RuntimeError(f"No test rows found in {test_path}")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    base_model = config["base_model"]

    print(f"Loading embedding model: {base_model}")
    embedder = SentenceTransformer(base_model)
    clf = joblib.load(model_dir / "classifier.joblib")
    transition_log_probs = np.load(model_dir / "transition_matrix.npy")

    print(f"Computing features for {len(test_rows)} test rows...")
    X_test, y_test = compute_features(test_rows, embedder)
    proba = align_proba_to_labels(clf.predict_proba(X_test), clf.classes_, LABELS)
    emission_log_probs = np.log(np.clip(proba, 1e-12, None))

    by_resume = _group_by_resume(test_rows)

    y_pred_no_smoothing: list[str] = []
    y_pred_smoothed: list[str] = []
    y_true: list[str] = []

    boundary_correct = 0
    boundary_total = 0

    row_index_of: dict[tuple[str, int], int] = {
        (r["resume_id"], r["line_index"]): i for i, r in enumerate(test_rows)
    }

    for resume_id, resume_rows in by_resume.items():
        idxs = [row_index_of[(resume_id, r["line_index"])] for r in resume_rows]
        resume_emissions = emission_log_probs[idxs]
        true_labels = [r["label"] for r in resume_rows]

        no_smoothing_labels = [LABELS[i] for i in np.argmax(resume_emissions, axis=1)]
        smoothed_indices = viterbi_decode(resume_emissions, transition_log_probs)
        smoothed_labels = [LABELS[i] for i in smoothed_indices]

        y_true.extend(true_labels)
        y_pred_no_smoothing.extend(no_smoothing_labels)
        y_pred_smoothed.extend(smoothed_labels)

        n_correct, n_positions = boundary_accuracy(true_labels, smoothed_labels)
        boundary_correct += n_correct
        boundary_total += n_positions

    print("\n=== Per-line report WITHOUT Viterbi smoothing ===")
    print(classification_report(y_true, y_pred_no_smoothing, labels=list(LABELS), zero_division=0))

    print("=== Per-line report WITH Viterbi smoothing ===")
    print(classification_report(y_true, y_pred_smoothed, labels=list(LABELS), zero_division=0))

    boundary_pct = boundary_correct / boundary_total * 100 if boundary_total else 0.0
    print(f"Boundary accuracy (section-change detection, Viterbi-smoothed): "
          f"{boundary_correct}/{boundary_total} ({boundary_pct:.1f}%)")
    print(f"Resumes evaluated: {len(by_resume)}   Total lines: {len(test_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the section-classifier on a held-out test split.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(test_path=args.test, model_dir=args.model)


if __name__ == "__main__":
    main()
