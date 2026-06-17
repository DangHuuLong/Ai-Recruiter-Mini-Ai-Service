"""
Fine-tune a cross-encoder on CV-JD pairs (regression, label = score/100).

Reads formatted JSONL files produced by training/prepare_cross_encoder_data.py.
Uses MSE loss with sigmoid activation so the model outputs a calibrated 0-1 score.
After training, predicted_score = model.predict(pair) * 100.

Usage:
  python -m training.fine_tune_cross_encoder
  python -m training.fine_tune_cross_encoder \\
      --data-dir datasets/versions/v0.3/cross_encoder \\
      --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
      --output-dir artifacts/models/cross-encoder-cv-jd-v0.1 \\
      --epochs 10 --batch-size 16

Debug (quick sanity check):
  python -m training.fine_tune_cross_encoder \\
      --epochs 1 --batch-size 4 \\
      --max-train-samples 40 --max-eval-samples 20
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


# ── label helpers (mirrors evaluate_similarity_pipeline.py) ──────────────────

def _label_for_score(score: float) -> str:
    if score >= 90.0:
        return "excellent_match"
    if score >= 75.0:
        return "strong_match"
    if score >= 60.0:
        return "moderate_match"
    if score >= 40.0:
        return "weak_match"
    return "poor_match"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_examples(path: Path, max_samples: int | None = None):
    from sentence_transformers import InputExample

    examples: list[InputExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            examples.append(
                InputExample(texts=[d["cv_text"], d["jd_text"]], label=float(d["label"]))
            )
            if max_samples and len(examples) >= max_samples:
                break
    return examples


def _load_raw(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
            if max_samples and len(records) >= max_samples:
                break
    return records


# ── evaluation ────────────────────────────────────────────────────────────────

def _compute_metrics(
    predictions_0_1: list[float],
    true_scores_0_100: list[float],
) -> dict[str, float]:
    pred_100 = [p * 100 for p in predictions_0_1]
    n = len(pred_100)
    mae = sum(abs(p - t) for p, t in zip(pred_100, true_scores_0_100)) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(pred_100, true_scores_0_100)) / n)
    label_acc = sum(
        1 for p, t in zip(pred_100, true_scores_0_100)
        if _label_for_score(p) == _label_for_score(t)
    ) / n
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "label_accuracy": round(label_acc, 4),
        "pair_count": n,
        "mean_predicted_score": round(sum(pred_100) / n, 4),
        "mean_target_score": round(sum(true_scores_0_100) / n, 4),
    }


def _evaluate_split(
    model,
    data_dir: Path,
    split: str,
    max_samples: int | None = None,
) -> dict[str, float]:
    records = _load_raw(data_dir / f"cross_encoder_{split}.jsonl", max_samples)
    pairs = [(r["cv_text"], r["jd_text"]) for r in records]
    true_scores = [r["score"] for r in records]

    preds = model.predict(pairs, batch_size=32, show_progress_bar=True)
    return _compute_metrics(list(preds), true_scores)


# ── training ──────────────────────────────────────────────────────────────────

def fine_tune(
    data_dir: Path,
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    max_length: int,
    warmup_ratio: float,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    report_path: Path | None,
) -> None:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
    from torch.utils.data import DataLoader

    print(f"Base model : {base_model}")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Epochs     : {epochs}  |  Batch size: {batch_size}  |  Max length: {max_length}\n")

    train_examples = _load_examples(data_dir / "cross_encoder_train.jsonl", max_train_samples)
    val_examples = _load_examples(data_dir / "cross_encoder_validation.jsonl", max_eval_samples)
    print(f"Train: {len(train_examples)} pairs  |  Val: {len(val_examples)} pairs\n")

    model = CrossEncoder(
        base_model,
        num_labels=1,
        max_length=max_length,
        default_activation_function=torch.nn.Sigmoid(),
    )

    steps_per_epoch = math.ceil(len(train_examples) / batch_size)
    warmup_steps = int(steps_per_epoch * warmup_ratio)
    print(f"Steps/epoch: {steps_per_epoch}  |  Warmup steps: {warmup_steps}\n")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    evaluator = CECorrelationEvaluator.from_input_examples(val_examples, name="val")

    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=steps_per_epoch,
        save_best_model=True,
        output_path=str(output_dir),
        loss_fct=torch.nn.MSELoss(),
        activation_fct=torch.nn.Sigmoid(),
        show_progress_bar=True,
    )

    print("\nFine-tuning complete.")
    print(f"Model written to: {output_dir}")

    print("\nEvaluating on validation split…")
    val_metrics = _evaluate_split(model, data_dir, "validation", max_eval_samples)
    print(
        f"  MAE: {val_metrics['mae']:.4f}  RMSE: {val_metrics['rmse']:.4f}"
        f"  LabelAcc: {val_metrics['label_accuracy']:.4f}"
    )

    print("Evaluating on test split…")
    test_metrics = _evaluate_split(model, data_dir, "test", max_eval_samples)
    print(
        f"  MAE: {test_metrics['mae']:.4f}  RMSE: {test_metrics['rmse']:.4f}"
        f"  LabelAcc: {test_metrics['label_accuracy']:.4f}"
    )

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "base_model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "train_pairs": len(train_examples),
            "val_pairs": len(val_examples),
            "metrics": {
                "validation": val_metrics,
                "test": test_metrics,
            },
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Report written to: {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune cross-encoder for CV-JD regression scoring"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/versions/v0.3/cross_encoder"),
        dest="data_dir",
    )
    parser.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        dest="base_model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models/cross-encoder-cv-jd-v0.1"),
        dest="output_dir",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/reports/fine_tune_cross_encoder_v0.1_report.json"),
        dest="report_path",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("--max-length", type=int, default=512, dest="max_length")
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        dest="warmup_ratio",
        help="Fraction of total steps used for linear warmup",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        dest="max_train_samples",
        help="Limit train pairs (debug mode)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        dest="max_eval_samples",
        help="Limit val/test pairs (debug mode)",
    )
    args = parser.parse_args()

    fine_tune(
        data_dir=args.data_dir,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
