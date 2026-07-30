"""
Fit an isotonic regression calibrator on cross-encoder validation outputs.

Run this once after fine-tuning to produce calibrator.pkl alongside the model.
The calibrator maps raw cross-encoder scores (sigmoid(logit)×100) to the true
score distribution, improving label accuracy without retraining.

Workflow:
  1. python -m training.fine_tune_cross_encoder    # train model
  2. python -m training.calibrate_cross_encoder    # fit calibrator  ← this script
  3. python -m training.evaluate_cross_encoder_pipeline  # benchmark

Usage:
  python -m training.calibrate_cross_encoder
  python -m training.calibrate_cross_encoder \\
      --cross-encoder models/cross-encoder-cv-jd-v0.1 \\
      --data-dir datasets/versions/v0.3/cross_encoder \\
      --output models/cross-encoder-cv-jd-v0.1/calibrator.pkl
"""
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any

from training.evaluate_similarity_pipeline import compute_metrics


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def _load_raw(path: Path) -> list[dict[str, Any]]:
    import json

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _predict_scores(model: Any, records: list[dict[str, Any]], batch_size: int) -> list[float]:
    """Return sigmoid(logit)*100 for every record using a pre-loaded CrossEncoder."""
    pairs = [(r["cv_text"], r["jd_text"]) for r in records]
    print(f"  Scoring {len(pairs)} pairs…")
    logits = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return [_sigmoid(float(x)) * 100 for x in logits]


def _print_metrics_row(tag: str, preds: list[float], true_scores: list[float]) -> None:
    m = compute_metrics(preds, true_scores)
    print(
        f"  {tag:<24}  MAE={m.mae:.4f}  RMSE={m.rmse:.4f}"
        f"  LabelAcc={m.label_accuracy:.4%}"
    )


def calibrate(
    model_path: str,
    data_dir: Path,
    output_path: Path,
    max_length: int,
    batch_size: int,
) -> None:
    from sentence_transformers import CrossEncoder
    from sklearn.isotonic import IsotonicRegression

    import numpy as np

    print(f"Loading cross-encoder: {model_path}")
    model = CrossEncoder(model_path, max_length=max_length)

    # ── Validation: fit calibrator ────────────────────────────────────────────
    print("\nLoading validation data…")
    val_records = _load_raw(data_dir / "cross_encoder_validation.jsonl")
    print(f"  {len(val_records)} pairs")

    val_raw = _predict_scores(model, val_records, batch_size)
    val_true = [r["score"] for r in val_records]

    print("\nFitting isotonic calibrator on validation set…")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw, val_true)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved → {output_path}\n")

    val_cal = calibrator.predict(np.array(val_raw)).tolist()
    print("── Validation ────────────────────────────────────────────")
    _print_metrics_row("raw (sigmoid×100)", val_raw, val_true)
    _print_metrics_row("calibrated", val_cal, val_true)

    # ── Test: out-of-sample check ─────────────────────────────────────────────
    print("\nLoading test data…")
    test_records = _load_raw(data_dir / "cross_encoder_test.jsonl")
    print(f"  {len(test_records)} pairs")

    test_raw = _predict_scores(model, test_records, batch_size)
    test_true = [r["score"] for r in test_records]
    test_cal = calibrator.predict(np.array(test_raw)).tolist()

    print("\n── Test ──────────────────────────────────────────────────")
    _print_metrics_row("raw (sigmoid×100)", test_raw, test_true)
    _print_metrics_row("calibrated", test_cal, test_true)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit isotonic regression calibrator for a fine-tuned cross-encoder"
    )
    parser.add_argument(
        "--cross-encoder",
        default="models/cross-encoder-cv-jd-v0.1",
        dest="cross_encoder",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/versions/v0.3/cross_encoder"),
        dest="data_dir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for calibrator.pkl (default: <model_dir>/calibrator.pkl)",
    )
    parser.add_argument("--max-length", type=int, default=512, dest="max_length")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    args = parser.parse_args()

    output = args.output or Path(args.cross_encoder) / "calibrator.pkl"

    calibrate(
        model_path=args.cross_encoder,
        data_dir=args.data_dir,
        output_path=output,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
