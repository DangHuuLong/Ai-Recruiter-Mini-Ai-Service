"""Fine-tune a SentenceTransformer model for CV-JD similarity.

This script is designed to run on Google Colab or Kaggle with GPU support.
It loads a versioned CV-JD dataset snapshot, builds paired resume/JD texts,
fine-tunes a SentenceTransformer model with CosineSimilarityLoss, evaluates
on validation and test splits, and writes a lightweight JSON report.

Example:
    python training/fine_tune_similarity.py \
      --dataset-root datasets \
      --version v0.2 \
      --base-model sentence-transformers/all-MiniLM-L6-v2 \
      --output-dir artifacts/models/fine-tuned-miniLM-v0.2 \
      --report-path artifacts/reports/fine_tune_similarity_v0.2_report.json \
      --epochs 1 \
      --batch-size 8

For a quick local smoke test, limit the number of samples:

    python training/fine_tune_similarity.py \
      --dataset-root datasets \
      --version v0.2 \
      --output-dir artifacts/models/debug-miniLM-v0.2 \
      --report-path artifacts/reports/debug_fine_tune_similarity_v0.2_report.json \
      --epochs 1 \
      --batch-size 2 \
      --max-train-samples 20 \
      --max-eval-samples 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sentence_transformers import InputExample, SentenceTransformer, losses
except ImportError:  # pragma: no cover - exercised manually when dependency is missing
    InputExample = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]
    losses = None  # type: ignore[assignment]

try:
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - exercised manually when dependency is missing
    DataLoader = None  # type: ignore[assignment]


DEFAULT_DATASET_ROOT = "datasets"
DEFAULT_VERSION = "v0.2"
DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_DIR = "artifacts/models/fine-tuned-miniLM-v0.2"
DEFAULT_REPORT_PATH = "artifacts/reports/fine_tune_similarity_v0.2_report.json"
BASELINE_V0_2 = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "pairs_evaluated": 2275,
    "mae": 17.6431,
    "rmse": 22.4497,
    "label_accuracy": 0.2822,
}


@dataclass(frozen=True)
class PairExample:
    pair_id: str
    resume_id: str
    job_description_id: str
    split: str
    label: str
    target_score: int
    resume_text: str
    job_description_text: str


@dataclass(frozen=True)
class PairPrediction:
    pair_id: str
    split: str
    label: str
    target_score: int
    similarity: float
    predicted_score: int
    absolute_error: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc.msg}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")

        records.append(data)

    return records


def resolve_dataset_paths(dataset_root: Path, version: str) -> tuple[Path, Path, Path]:
    base_path = dataset_root / "versions" / version
    return (
        base_path / "job_descriptions.jsonl",
        base_path / "resumes.jsonl",
        base_path / "cv_jd_pairs.jsonl",
    )


def list_text(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        rendered: list[str] = []
        for item in values:
            if isinstance(item, str):
                rendered.append(item)
            elif isinstance(item, dict):
                rendered.extend(str(value) for value in item.values() if isinstance(value, (str, int, float)))
        return "; ".join(rendered)
    if isinstance(values, dict):
        return "; ".join(str(value) for value in values.values() if isinstance(value, (str, int, float)))
    return str(values)


def build_resume_text(resume: dict[str, Any]) -> str:
    sections = [
        f"Summary: {resume.get('summary', '')}",
        f"Skills: {list_text(resume.get('skills'))}",
        f"Normalized skills: {list_text(resume.get('normalized_skills'))}",
        f"Experience years: {resume.get('experience_years', '')}",
        f"Education: {list_text(resume.get('education'))}",
        f"Projects: {list_text(resume.get('projects'))}",
        f"Raw text: {resume.get('raw_text', '')}",
    ]
    return "\n".join(section for section in sections if section.strip())


def build_job_description_text(jd: dict[str, Any]) -> str:
    sections = [
        f"Title: {jd.get('title', '')}",
        f"Level: {jd.get('level', '')}",
        f"Domain: {jd.get('domain', '')}",
        f"Responsibilities: {list_text(jd.get('responsibilities'))}",
        f"Requirements: {list_text(jd.get('requirements'))}",
        f"Required skills: {list_text(jd.get('required_skills'))}",
        f"Preferred skills: {list_text(jd.get('preferred_skills'))}",
        f"Minimum experience years: {jd.get('min_experience_years', '')}",
        f"Education requirement: {jd.get('education_requirement', '')}",
        f"Domain keywords: {list_text(jd.get('domain_keywords'))}",
        f"Raw text: {jd.get('raw_text', '')}",
    ]
    return "\n".join(section for section in sections if section.strip())


def label_from_score(score: int) -> str:
    if score >= 90:
        return "excellent_match"
    if score >= 75:
        return "strong_match"
    if score >= 60:
        return "moderate_match"
    if score >= 40:
        return "weak_match"
    return "poor_match"


def similarity_to_score(similarity: float) -> int:
    normalized = (similarity + 1.0) / 2.0
    clipped = max(0.0, min(1.0, normalized))
    return round(clipped * 100)


def load_pair_examples(dataset_paths: tuple[Path, Path, Path]) -> list[PairExample]:
    jd_records = read_jsonl(dataset_paths[0])
    resume_records = read_jsonl(dataset_paths[1])
    pair_records = read_jsonl(dataset_paths[2])

    jd_by_id = {str(record.get("id")): record for record in jd_records}
    resume_by_id = {str(record.get("id")): record for record in resume_records}

    examples: list[PairExample] = []
    for pair in pair_records:
        pair_id = str(pair.get("id"))
        resume_id = str(pair.get("resume_id"))
        jd_id = str(pair.get("job_description_id"))
        split = pair.get("split")
        label = str(pair.get("label"))
        target_score = pair.get("overall_score")

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Pair {pair_id} has invalid split: {split!r}")
        if not isinstance(target_score, int):
            raise ValueError(f"Pair {pair_id} has invalid overall_score: {target_score!r}")
        if resume_id not in resume_by_id:
            raise ValueError(f"Pair {pair_id} references missing resume_id: {resume_id}")
        if jd_id not in jd_by_id:
            raise ValueError(f"Pair {pair_id} references missing job_description_id: {jd_id}")

        examples.append(
            PairExample(
                pair_id=pair_id,
                resume_id=resume_id,
                job_description_id=jd_id,
                split=str(split),
                label=label,
                target_score=target_score,
                resume_text=build_resume_text(resume_by_id[resume_id]),
                job_description_text=build_job_description_text(jd_by_id[jd_id]),
            )
        )

    return examples


def group_by_split(examples: list[PairExample]) -> dict[str, list[PairExample]]:
    grouped = {"train": [], "validation": [], "test": []}
    for example in examples:
        grouped[example.split].append(example)
    return grouped


def limit_examples(examples: list[PairExample], limit: int | None, seed: int) -> list[PairExample]:
    if limit is None or limit <= 0 or len(examples) <= limit:
        return examples

    rng = random.Random(seed)
    selected = list(examples)
    rng.shuffle(selected)
    return selected[:limit]


def to_input_examples(examples: list[PairExample]) -> list[Any]:
    if InputExample is None:
        raise RuntimeError("Missing dependency: sentence-transformers")

    return [
        InputExample(
            texts=[example.resume_text, example.job_description_text],
            label=example.target_score / 100.0,
        )
        for example in examples
    ]


def evaluate_model(model: Any, examples: list[PairExample], batch_size: int) -> list[PairPrediction]:
    predictions: list[PairPrediction] = []
    if not examples:
        return predictions

    resume_embeddings = model.encode(
        [example.resume_text for example in examples],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    jd_embeddings = model.encode(
        [example.job_description_text for example in examples],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for example, resume_embedding, jd_embedding in zip(examples, resume_embeddings, jd_embeddings):
        similarity = float(sum(float(a) * float(b) for a, b in zip(resume_embedding, jd_embedding)))
        predicted_score = similarity_to_score(similarity)
        predictions.append(
            PairPrediction(
                pair_id=example.pair_id,
                split=example.split,
                label=example.label,
                target_score=example.target_score,
                similarity=similarity,
                predicted_score=predicted_score,
                absolute_error=abs(predicted_score - example.target_score),
            )
        )

    return predictions


def calculate_metrics(predictions: list[PairPrediction]) -> dict[str, Any]:
    if not predictions:
        return {
            "pair_count": 0,
            "mae": None,
            "rmse": None,
            "label_accuracy": None,
        }

    absolute_errors = [prediction.absolute_error for prediction in predictions]
    squared_errors = [error * error for error in absolute_errors]
    label_matches = [label_from_score(prediction.predicted_score) == prediction.label for prediction in predictions]

    return {
        "pair_count": len(predictions),
        "mae": round(statistics.mean(absolute_errors), 4),
        "rmse": round(math.sqrt(statistics.mean(squared_errors)), 4),
        "label_accuracy": round(sum(label_matches) / len(label_matches), 4),
        "mean_similarity": round(statistics.mean(prediction.similarity for prediction in predictions), 6),
        "mean_predicted_score": round(statistics.mean(prediction.predicted_score for prediction in predictions), 4),
        "mean_target_score": round(statistics.mean(prediction.target_score for prediction in predictions), 4),
    }


def write_report(
    report_path: Path,
    dataset_paths: tuple[Path, Path, Path],
    args: argparse.Namespace,
    split_counts: dict[str, int],
    validation_predictions: list[PairPrediction],
    test_predictions: list[PairPrediction],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": {
            "dataset_version": args.version,
            "base_model": args.base_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
        },
        "dataset": {
            "job_descriptions_path": str(dataset_paths[0]),
            "resumes_path": str(dataset_paths[1]),
            "pairs_path": str(dataset_paths[2]),
            "split_counts": split_counts,
        },
        "baseline_v0_2": BASELINE_V0_2,
        "metrics": {
            "validation": calculate_metrics(validation_predictions),
            "test": calculate_metrics(test_predictions),
        },
        "predictions": {
            "validation": [prediction.__dict__ for prediction in validation_predictions],
            "test": [prediction.__dict__ for prediction in test_predictions],
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a CV-JD SentenceTransformer similarity model.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Dataset root directory. Defaults to datasets.")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Versioned dataset snapshot. Defaults to v0.2.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base SentenceTransformer model name.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Model output directory.")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, help="Fine-tuning report output path.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs. Defaults to 1.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training and evaluation batch size. Defaults to 8.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate. Defaults to 2e-5.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio based on train steps. Defaults to 0.1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Defaults to 42.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for quick local smoke tests.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional cap for validation/test smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if SentenceTransformer is None or losses is None or DataLoader is None:
            raise RuntimeError(
                "Missing dependencies. Install sentence-transformers and torch before running fine-tuning."
            )

        random.seed(args.seed)

        dataset_paths = resolve_dataset_paths(Path(args.dataset_root), args.version)
        examples = load_pair_examples(dataset_paths)
        by_split = group_by_split(examples)

        train_examples = limit_examples(by_split["train"], args.max_train_samples, args.seed)
        validation_examples = limit_examples(by_split["validation"], args.max_eval_samples, args.seed)
        test_examples = limit_examples(by_split["test"], args.max_eval_samples, args.seed)

        if not train_examples:
            raise ValueError("No train examples found. Check split assignments in cv_jd_pairs.jsonl.")
        if not validation_examples:
            raise ValueError("No validation examples found. Check split assignments in cv_jd_pairs.jsonl.")
        if not test_examples:
            raise ValueError("No test examples found. Check split assignments in cv_jd_pairs.jsonl.")

        model = SentenceTransformer(args.base_model)
        train_dataloader = DataLoader(
            to_input_examples(train_examples),
            shuffle=True,
            batch_size=args.batch_size,
        )
        train_loss = losses.CosineSimilarityLoss(model=model)

        train_steps = len(train_dataloader) * args.epochs
        warmup_steps = round(train_steps * args.warmup_ratio)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Fine-tuning CV-JD similarity model.")
        print(f"Dataset version: {args.version}")
        print(f"Base model: {args.base_model}")
        print(f"Train examples: {len(train_examples)}")
        print(f"Validation examples: {len(validation_examples)}")
        print(f"Test examples: {len(test_examples)}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Warmup steps: {warmup_steps}")

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.learning_rate},
            output_path=str(output_dir),
            show_progress_bar=True,
        )

        tuned_model = SentenceTransformer(str(output_dir))
        validation_predictions = evaluate_model(tuned_model, validation_examples, args.batch_size)
        test_predictions = evaluate_model(tuned_model, test_examples, args.batch_size)

        report_path = Path(args.report_path)
        split_counts = {split: len(records) for split, records in by_split.items()}
        write_report(
            report_path=report_path,
            dataset_paths=dataset_paths,
            args=args,
            split_counts=split_counts,
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
        )

        validation_metrics = calculate_metrics(validation_predictions)
        test_metrics = calculate_metrics(test_predictions)

        print("Fine-tuning complete.")
        print(f"Model written to: {output_dir}")
        print(f"Report written to: {report_path}")
        print("Validation metrics:")
        print(f"  MAE: {validation_metrics['mae']}")
        print(f"  RMSE: {validation_metrics['rmse']}")
        print(f"  Label accuracy: {validation_metrics['label_accuracy']}")
        print("Test metrics:")
        print(f"  MAE: {test_metrics['mae']}")
        print(f"  RMSE: {test_metrics['rmse']}")
        print(f"  Label accuracy: {test_metrics['label_accuracy']}")
        return 0
    except Exception as exc:  # pragma: no cover - command-line guard
        print(f"Fine-tuning failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
