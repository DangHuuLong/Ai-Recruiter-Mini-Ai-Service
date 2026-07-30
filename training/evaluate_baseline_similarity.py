"""Evaluate a pretrained embedding baseline for CV-JD similarity.

This script loads the selected baseline model from training/baseline_models.json,
embeds resume and job description text, calculates cosine similarity for each
labeled CV-JD pair, and writes a lightweight JSON report.

Usage:
    python training/evaluate_baseline_similarity.py
    python training/evaluate_baseline_similarity.py --version v0.1
    python training/evaluate_baseline_similarity.py --model-key multilingual_mini_lm_l12_v2
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised manually when dependency is missing
    SentenceTransformer = None  # type: ignore[assignment]


DEFAULT_DATASET_ROOT = "datasets"
DEFAULT_MODEL_CONFIG_PATH = "training/baseline_models.json"
DEFAULT_OUTPUT_PATH = "artifacts/reports/baseline_similarity_report.json"


@dataclass(frozen=True)
class BaselineModelConfig:
    key: str
    model_name: str
    embedding_dimension: int | None
    max_sequence_length: int | None


@dataclass(frozen=True)
class PairPrediction:
    pair_id: str
    resume_id: str
    job_description_id: str
    label: str
    target_score: int
    similarity: float
    predicted_score: int
    absolute_error: int


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
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


def load_model_config(path: Path, model_key: str | None) -> BaselineModelConfig:
    data = read_json_file(path)

    selected = data.get("selected_primary_model")
    candidates = data.get("comparison_candidates", [])

    all_models: list[dict[str, Any]] = []
    if isinstance(selected, dict):
        all_models.append(selected)
    if isinstance(candidates, list):
        all_models.extend(candidate for candidate in candidates if isinstance(candidate, dict))

    if not all_models:
        raise ValueError(f"No baseline models configured in {path}")

    if model_key is None:
        if not isinstance(selected, dict):
            raise ValueError(f"selected_primary_model is missing or invalid in {path}")
        model_data = selected
    else:
        matches = [model for model in all_models if model.get("key") == model_key]
        if not matches:
            available = ", ".join(str(model.get("key")) for model in all_models)
            raise ValueError(f"Unknown model key {model_key!r}. Available keys: {available}")
        model_data = matches[0]

    key = model_data.get("key")
    model_name = model_data.get("model_name")
    if not isinstance(key, str) or not key:
        raise ValueError("Selected model config is missing key")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Selected model config is missing model_name")

    embedding_dimension = model_data.get("embedding_dimension")
    max_sequence_length = model_data.get("max_sequence_length")

    return BaselineModelConfig(
        key=key,
        model_name=model_name,
        embedding_dimension=embedding_dimension if isinstance(embedding_dimension, int) else None,
        max_sequence_length=max_sequence_length if isinstance(max_sequence_length, int) else None,
    )


def resolve_dataset_paths(dataset_root: Path, version: str | None) -> tuple[Path, Path, Path]:
    if version:
        base_path = dataset_root / "versions" / version
        return (
            base_path / "job_descriptions.jsonl",
            base_path / "resumes.jsonl",
            base_path / "cv_jd_pairs.jsonl",
        )

    return (
        dataset_root / "raw" / "job_descriptions.jsonl",
        dataset_root / "raw" / "resumes.jsonl",
        dataset_root / "processed" / "cv_jd_pairs.jsonl",
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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same length")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def similarity_to_score(similarity: float) -> int:
    """Map cosine similarity from [-1, 1] to [0, 100] for reporting.

    This is not calibrated. It is only a simple reporting scale for the first
    baseline evaluation.
    """

    normalized = (similarity + 1.0) / 2.0
    clipped = max(0.0, min(1.0, normalized))
    return round(clipped * 100)


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
    label_matches = [
        label_from_score(prediction.predicted_score) == prediction.label
        for prediction in predictions
    ]

    return {
        "pair_count": len(predictions),
        "mae": round(statistics.mean(absolute_errors), 4),
        "rmse": round(math.sqrt(statistics.mean(squared_errors)), 4),
        "label_accuracy": round(sum(label_matches) / len(label_matches), 4),
        "mean_similarity": round(statistics.mean(prediction.similarity for prediction in predictions), 6),
        "mean_predicted_score": round(statistics.mean(prediction.predicted_score for prediction in predictions), 4),
        "mean_target_score": round(statistics.mean(prediction.target_score for prediction in predictions), 4),
    }


def run_evaluation(
    model_config: BaselineModelConfig,
    jd_records: list[dict[str, Any]],
    resume_records: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
) -> list[PairPrediction]:
    if SentenceTransformer is None:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install dependencies with "
            "`pip install -r requirements.txt` before running baseline evaluation."
        )

    jd_by_id = {str(record.get("id")): record for record in jd_records}
    resume_by_id = {str(record.get("id")): record for record in resume_records}

    missing_resume_ids = sorted(
        str(pair.get("resume_id")) for pair in pair_records if str(pair.get("resume_id")) not in resume_by_id
    )
    missing_jd_ids = sorted(
        str(pair.get("job_description_id")) for pair in pair_records if str(pair.get("job_description_id")) not in jd_by_id
    )

    if missing_resume_ids:
        raise ValueError(f"Pair records reference missing resume IDs: {missing_resume_ids}")
    if missing_jd_ids:
        raise ValueError(f"Pair records reference missing job description IDs: {missing_jd_ids}")

    model = SentenceTransformer(model_config.model_name)

    resume_texts = {resume_id: build_resume_text(resume) for resume_id, resume in resume_by_id.items()}
    jd_texts = {jd_id: build_job_description_text(jd) for jd_id, jd in jd_by_id.items()}

    resume_embeddings = {
        resume_id: embedding.tolist()
        for resume_id, embedding in zip(
            resume_texts,
            model.encode(list(resume_texts.values()), normalize_embeddings=False),
        )
    }
    jd_embeddings = {
        jd_id: embedding.tolist()
        for jd_id, embedding in zip(
            jd_texts,
            model.encode(list(jd_texts.values()), normalize_embeddings=False),
        )
    }

    predictions: list[PairPrediction] = []
    for pair in pair_records:
        pair_id = str(pair.get("id"))
        resume_id = str(pair.get("resume_id"))
        jd_id = str(pair.get("job_description_id"))
        label = str(pair.get("label"))
        target_score = pair.get("overall_score")

        if not isinstance(target_score, int):
            raise ValueError(f"Pair {pair_id} has invalid overall_score: {target_score!r}")

        similarity = cosine_similarity(resume_embeddings[resume_id], jd_embeddings[jd_id])
        predicted_score = similarity_to_score(similarity)
        absolute_error = abs(predicted_score - target_score)

        predictions.append(
            PairPrediction(
                pair_id=pair_id,
                resume_id=resume_id,
                job_description_id=jd_id,
                label=label,
                target_score=target_score,
                similarity=similarity,
                predicted_score=predicted_score,
                absolute_error=absolute_error,
            )
        )

    return predictions


def write_report(
    output_path: Path,
    model_config: BaselineModelConfig,
    dataset_paths: tuple[Path, Path, Path],
    predictions: list[PairPrediction],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model": {
            "key": model_config.key,
            "model_name": model_config.model_name,
            "embedding_dimension": model_config.embedding_dimension,
            "max_sequence_length": model_config.max_sequence_length,
        },
        "dataset": {
            "job_descriptions_path": str(dataset_paths[0]),
            "resumes_path": str(dataset_paths[1]),
            "pairs_path": str(dataset_paths[2]),
        },
        "metrics": calculate_metrics(predictions),
        "predictions": [
            {
                "pair_id": prediction.pair_id,
                "resume_id": prediction.resume_id,
                "job_description_id": prediction.job_description_id,
                "label": prediction.label,
                "target_score": prediction.target_score,
                "similarity": round(prediction.similarity, 6),
                "predicted_score": prediction.predicted_score,
                "predicted_label": label_from_score(prediction.predicted_score),
                "absolute_error": prediction.absolute_error,
            }
            for prediction in predictions
        ],
    }

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline CV-JD embedding similarity.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Dataset root directory. Defaults to datasets.")
    parser.add_argument("--version", default=None, help="Evaluate a versioned dataset snapshot, for example v0.1.")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG_PATH, help="Baseline model config JSON path.")
    parser.add_argument("--model-key", default=None, help="Optional model key from baseline_models.json. Defaults to selected primary model.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        model_config = load_model_config(Path(args.model_config), args.model_key)
        dataset_paths = resolve_dataset_paths(Path(args.dataset_root), args.version)

        jd_records = read_jsonl(dataset_paths[0])
        resume_records = read_jsonl(dataset_paths[1])
        pair_records = read_jsonl(dataset_paths[2])

        predictions = run_evaluation(model_config, jd_records, resume_records, pair_records)
        metrics = calculate_metrics(predictions)
        output_path = Path(args.output)
        write_report(output_path, model_config, dataset_paths, predictions)

        print("Baseline similarity evaluation complete.")
        print(f"Model: {model_config.model_name}")
        print(f"Pairs evaluated: {metrics['pair_count']}")
        print(f"MAE: {metrics['mae']}")
        print(f"RMSE: {metrics['rmse']}")
        print(f"Label accuracy: {metrics['label_accuracy']}")
        print(f"Report written to: {output_path}")
        return 0
    except Exception as exc:  # pragma: no cover - command-line guard
        print(f"Baseline similarity evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
