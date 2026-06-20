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
import torch.nn.functional as F


# ── loss functions ────────────────────────────────────────────────────────────

class BoundaryAwareLoss(torch.nn.Module):
    """MSE + ordinal BCE at the 5-bucket boundaries (40/60/75/90).

    preds and labels are both in [0, 1] because activation_fct=Sigmoid is
    applied by CrossEncoder.fit() before this loss receives them.
    The ordinal component penalises predictions that land on the wrong side
    of a rubric boundary, directly targeting label accuracy.
    """
    _BOUNDARIES = [0.40, 0.60, 0.75, 0.90]
    _ALPHA = 0.3  # weight of ordinal term relative to MSE

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(preds, labels)
        ordinal = preds.new_zeros(1)
        for b in self._BOUNDARIES:
            target = (labels >= b).float()
            pred_logit = 20.0 * (preds - b)
            ordinal = ordinal + F.binary_cross_entropy_with_logits(pred_logit, target)
        return mse + self._ALPHA * ordinal / len(self._BOUNDARIES)


class CELabelAccEvaluator:
    def __init__(self, sentence_pairs: list, labels_0_100: list[float], name: str = ""):
        self.sentence_pairs = sentence_pairs
        self.labels_0_100 = labels_0_100
        self.name = name

    @classmethod
    def from_input_examples(cls, examples, name: str = "") -> "CELabelAccEvaluator":
        return cls(
            sentence_pairs=[ex.texts for ex in examples],
            labels_0_100=[ex.label * 100 for ex in examples],
            name=name,
        )

    def __call__(self, model, output_path=None, epoch: int = -1, steps: int = -1) -> float:
        import numpy as _np
        preds = model.predict(self.sentence_pairs, batch_size=32, show_progress_bar=False)
        arr = _np.asarray(preds)
        if arr.ndim == 2:
            pred_classes = arr.argmax(axis=1).tolist()
            true_classes = [_score_to_class_index(t) for t in self.labels_0_100]
            return sum(1 for p, t in zip(pred_classes, true_classes) if p == t) / len(pred_classes)
        pred_100 = [float(p) * 100 for p in preds]
        return sum(
            1 for p, t in zip(pred_100, self.labels_0_100)
            if _label_for_score(p) == _label_for_score(t)
        ) / len(pred_100)


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


def _score_to_class_index(score_0_100: float) -> int:
    if score_0_100 >= 90.0:
        return 4
    if score_0_100 >= 75.0:
        return 3
    if score_0_100 >= 60.0:
        return 2
    if score_0_100 >= 40.0:
        return 1
    return 0


_CLASS_CENTERS = [20.0, 50.0, 67.5, 82.5, 95.0]


class ClassificationLoss(torch.nn.Module):
    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_targets = torch.tensor(
            [_score_to_class_index(float(l) * 100) for l in labels],
            dtype=torch.long,
            device=preds.device,
        )
        return F.cross_entropy(preds, class_targets)


# ── data loading ──────────────────────────────────────────────────────────────

def _load_examples(path: Path, max_samples: int | None = None, class_indices: bool = False):
    from sentence_transformers import InputExample

    examples: list[InputExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            # classification: store class index (0–4) so internal CE loss receives correct long targets
            label = float(_score_to_class_index(float(d["score"]))) if class_indices else float(d["label"])
            examples.append(InputExample(texts=[d["cv_text"], d["jd_text"]], label=label))
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

def _preds_to_score_100(preds) -> list[float]:
    import numpy as _np
    arr = _np.asarray(preds)
    if arr.ndim == 2:
        return (_np.array(_CLASS_CENTERS) @ arr.T).tolist()
    return (arr * 100).tolist()


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
    scores_0_1 = [s / 100 for s in _preds_to_score_100(preds)]
    return _compute_metrics(scores_0_1, true_scores)


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
    loss_type: str = "mse",
    evaluator_type: str = "spearman",
) -> None:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
    from torch.utils.data import DataLoader

    is_classification = loss_type == "classification"
    if loss_type == "classification":
        # loss_fct=None: use sentence-transformers' internal CrossEntropyLoss for num_labels=5.
        # Labels must be integer class indices (0–4 stored as float) so .long() gives correct targets.
        loss_fct = None
    elif loss_type == "boundary":
        loss_fct = BoundaryAwareLoss()
    else:
        loss_fct = torch.nn.MSELoss()

    print(f"Base model : {base_model}")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    loss_label = "CrossEntropyLoss (internal)" if is_classification else loss_fct.__class__.__name__
    print(f"Loss       : {loss_type}  ({loss_label})")
    print(f"Evaluator  : {evaluator_type}")
    print(f"Epochs     : {epochs}  |  Batch size: {batch_size}  |  Max length: {max_length}\n")

    train_examples = _load_examples(data_dir / "cross_encoder_train.jsonl", max_train_samples, class_indices=is_classification)
    val_examples = _load_examples(data_dir / "cross_encoder_validation.jsonl", max_eval_samples, class_indices=is_classification)
    print(f"Train: {len(train_examples)} pairs  |  Val: {len(val_examples)} pairs\n")

    model = CrossEncoder(
        base_model,
        num_labels=5 if is_classification else 1,
        max_length=max_length,
        default_activation_function=torch.nn.Softmax(dim=-1) if is_classification else torch.nn.Sigmoid(),
        automodel_args={"ignore_mismatched_sizes": True},
    )

    steps_per_epoch = math.ceil(len(train_examples) / batch_size)
    warmup_steps = int(steps_per_epoch * warmup_ratio)
    print(f"Steps/epoch: {steps_per_epoch}  |  Warmup steps: {warmup_steps}\n")

    from torch.utils.data import WeightedRandomSampler

    _LABEL_WEIGHT = {
        "poor_match":      1.0,
        "weak_match":      1.38,
        "moderate_match":  3.50,
        "strong_match":    3.93,
        "excellent_match": 2.78,
    }
    sample_weights = [
        _LABEL_WEIGHT.get(_label_for_score(ex.label * 100), 1.0)
        for ex in train_examples
    ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(train_examples, sampler=sampler, batch_size=batch_size)

    if evaluator_type == "label_acc":
        evaluator = CELabelAccEvaluator.from_input_examples(val_examples, name="val")
    else:
        evaluator = CECorrelationEvaluator.from_input_examples(val_examples, name="val")

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_metric = "label_acc" if evaluator_type == "label_acc" else "spearman"

    def _epoch_callback(score: float, epoch: int, steps: int) -> None:
        if steps == -1:
            print(f"  [done]  val_{eval_metric}={score:.4f}", flush=True)
        else:
            print(f"  epoch {epoch + 1:>2}/{epochs}  step {steps:>5}  val_{eval_metric}={score:.4f}", flush=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=steps_per_epoch,
        save_best_model=True,
        output_path=str(output_dir),
        loss_fct=loss_fct,
        activation_fct=torch.nn.Identity() if is_classification else torch.nn.Sigmoid(),
        use_amp=True,
        show_progress_bar=False,
        callback=_epoch_callback,
    )

    print("\nFine-tuning complete.")
    print(f"Model written to: {output_dir}")

    # Reload best checkpoint — fit() leaves in-memory model at last epoch
    best_model = CrossEncoder(
        str(output_dir),
        num_labels=5 if is_classification else 1,
        max_length=max_length,
        default_activation_function=torch.nn.Softmax(dim=-1) if is_classification else torch.nn.Sigmoid(),
    )

    print("\nEvaluating on validation split…")
    val_metrics = _evaluate_split(best_model, data_dir, "validation", max_eval_samples)
    print(
        f"  MAE: {val_metrics['mae']:.4f}  RMSE: {val_metrics['rmse']:.4f}"
        f"  LabelAcc: {val_metrics['label_accuracy']:.4f}"
    )

    print("Evaluating on test split…")
    test_metrics = _evaluate_split(best_model, data_dir, "test", max_eval_samples)
    print(
        f"  MAE: {test_metrics['mae']:.4f}  RMSE: {test_metrics['rmse']:.4f}"
        f"  LabelAcc: {test_metrics['label_accuracy']:.4f}"
    )

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "base_model": base_model,
            "loss": loss_type,
            "evaluator": evaluator_type,
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
        "--loss",
        choices=["mse", "boundary", "classification"],
        default="mse",
        help="mse: regression MSELoss; boundary: MSE + ordinal BCE; classification: 5-class CrossEntropy",
    )
    parser.add_argument(
        "--evaluator",
        choices=["spearman", "label_acc"],
        default="spearman",
        dest="evaluator",
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
        loss_type=args.loss,
        evaluator_type=args.evaluator,
    )


if __name__ == "__main__":
    main()
