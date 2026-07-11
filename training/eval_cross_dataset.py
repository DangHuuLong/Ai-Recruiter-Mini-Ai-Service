"""
Evaluate a fine-tuned cross-encoder checkpoint against a DIFFERENT dataset's test
split than the one it was trained on — used to check whether a MAE/LabelAcc gain
is a genuine improvement or an artifact of the training dataset being easier.

Usage:
  python -m training.eval_cross_dataset
  python -m training.eval_cross_dataset --model models/cross-encoder-cv-jd-v0.7 \\
      --data-dir datasets/versions/v0.5/cross_encoder --split test

Prints MAE / RMSE / LabelAcc directly to the terminal. Writes no report files.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_THRESHOLDS: list[tuple[float, str]] = [
    (90.0, "excellent_match"),
    (75.0, "strong_match"),
    (60.0, "moderate_match"),
    (40.0, "weak_match"),
    (0.0, "poor_match"),
]


def _label_for_score(score: float) -> str:
    for threshold, label in _THRESHOLDS:
        if score >= threshold:
            return label
    return "poor_match"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/cross-encoder-cv-jd-v0.7"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/versions/v0.5/cross_encoder"))
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--max-length", type=int, default=512, dest="max_length")
    args = parser.parse_args()

    data_path = args.data_dir / f"cross_encoder_{args.split}.jsonl"
    records = _load_jsonl(data_path)
    if not records:
        print(f"[!] No records found at {data_path}")
        return

    pairs = [(r["cv_text"], r["jd_text"]) for r in records]
    true_scores = [float(r["score"]) for r in records]

    print(f"Model      : {args.model}")
    print(f"Eval data  : {data_path}  ({len(records)} pairs)")
    print("Loading model and running inference...")

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(str(args.model), max_length=args.max_length)
    preds_0_1 = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    pred_100 = [float(p) * 100 for p in preds_0_1]

    n = len(pred_100)
    mae = sum(abs(p - t) for p, t in zip(pred_100, true_scores)) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(pred_100, true_scores)) / n)
    label_acc = sum(
        1 for p, t in zip(pred_100, true_scores)
        if _label_for_score(p) == _label_for_score(t)
    ) / n

    print()
    print(f"Cross-dataset eval — model trained on one dataset, tested on another")
    print(f"  pair_count           : {n}")
    print(f"  mae                  : {round(mae, 4)}")
    print(f"  rmse                 : {round(rmse, 4)}")
    print(f"  label_accuracy       : {round(label_acc, 4)}  ({label_acc * 100:.2f}%)")
    print(f"  mean_predicted_score : {round(sum(pred_100) / n, 4)}")
    print(f"  mean_target_score    : {round(sum(true_scores) / n, 4)}")


if __name__ == "__main__":
    main()
