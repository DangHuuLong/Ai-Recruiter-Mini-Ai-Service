from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ── label helpers ─────────────────────────────────────────────────────────────

_THRESHOLDS: list[tuple[float, str]] = [
    (90.0, "excellent_match"),
    (75.0, "strong_match"),
    (60.0, "moderate_match"),
    (40.0, "weak_match"),
    (0.0,  "poor_match"),
]


def label_for_score(score: float) -> str:
    for threshold, label in _THRESHOLDS:
        if score >= threshold:
            return label
    return "poor_match"


# ── data loading ──────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    pair_id: str
    true_score: float
    true_label: str
    cv_text: str
    jd_text: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_resume_text(r: dict[str, Any]) -> str:
    parts: list[str] = []
    if r.get("summary"):
        parts.append(str(r["summary"]))
    for s in r.get("skills") or []:
        if isinstance(s, str):
            parts.append(s)
        elif isinstance(s, dict):
            parts.append(s.get("name", "") or "")
    for exp in r.get("experience") or []:
        if isinstance(exp, dict):
            parts.append(exp.get("role", "") or "")
            parts.append(exp.get("company", "") or "")
            for resp in exp.get("responsibilities") or []:
                parts.append(str(resp))
    for proj in r.get("projects") or []:
        if isinstance(proj, dict):
            parts.append(proj.get("name", "") or "")
            parts.append(proj.get("description", "") or "")
    for edu in r.get("education") or []:
        if isinstance(edu, dict):
            parts.append(edu.get("degree", "") or "")
            parts.append(edu.get("field", "") or "")
    return " ".join(p for p in parts if p.strip())


def _extract_jd_text(jd: dict[str, Any]) -> str:
    parts: list[str] = []
    if jd.get("title"):
        parts.append(str(jd["title"]))
    if jd.get("level"):
        parts.append(str(jd["level"]))
    for key in ("responsibilities", "requirements", "nice_to_have", "domain_keywords"):
        for item in jd.get(key) or []:
            parts.append(str(item))
    return " ".join(p for p in parts if p.strip())


def load_test_pairs(
    pairs_path: Path,
    resumes_path: Path,
    jds_path: Path,
) -> list[EvalSample]:
    resumes = {r["id"]: r for r in _load_jsonl(resumes_path)}
    jds = {j["id"]: j for j in _load_jsonl(jds_path)}
    samples: list[EvalSample] = []
    for pair in _load_jsonl(pairs_path):
        if pair.get("split") != "test":
            continue
        resume = resumes.get(pair["resume_id"])
        jd = jds.get(pair["job_description_id"])
        if resume is None or jd is None:
            continue
        true_score = float(pair["overall_score"])
        samples.append(EvalSample(
            pair_id=pair["id"],
            true_score=true_score,
            true_label=pair.get("label") or label_for_score(true_score),
            cv_text=_extract_resume_text(resume),
            jd_text=_extract_jd_text(jd),
        ))
    return samples


# ── metrics ───────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    mae: float
    rmse: float
    label_accuracy: float
    n_samples: int


def compute_metrics(predictions: list[float], true_scores: list[float]) -> Metrics:
    n = len(predictions)
    if n == 0:
        return Metrics(mae=0.0, rmse=0.0, label_accuracy=0.0, n_samples=0)
    mae = sum(abs(p - t) for p, t in zip(predictions, true_scores)) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(predictions, true_scores)) / n)
    label_acc = sum(
        1 for p, t in zip(predictions, true_scores)
        if label_for_score(p) == label_for_score(t)
    ) / n
    return Metrics(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        label_accuracy=round(label_acc, 4),
        n_samples=n,
    )


# ── model evaluation ──────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    model_path: str
    metrics: Metrics
    predictions: list[dict[str, Any]] = field(default_factory=list)


def _dot_product(a: list[float], b: list[float]) -> float:
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def run_evaluation(
    model_name: str,
    model_path: str,
    samples: list[EvalSample],
    batch_size: int = 64,
) -> ModelResult:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("sentence-transformers is not installed") from e

    print(f"  Loading: {model_path or model_name}")
    model = SentenceTransformer(model_path or model_name)

    all_texts = [s.cv_text for s in samples] + [s.jd_text for s in samples]
    print(f"  Encoding {len(samples) * 2} texts…")
    all_embs = model.encode(
        all_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    cv_embs = all_embs[: len(samples)]
    jd_embs = all_embs[len(samples) :]

    predictions: list[float] = []
    prediction_records: list[dict[str, Any]] = []
    for sample, cv_emb, jd_emb in zip(samples, cv_embs, jd_embs):
        sim = _dot_product(list(cv_emb), list(jd_emb))
        pred_score = round(sim * 100, 2)
        predictions.append(pred_score)
        prediction_records.append({
            "pair_id": sample.pair_id,
            "true_score": sample.true_score,
            "predicted_score": pred_score,
            "true_label": sample.true_label,
            "predicted_label": label_for_score(pred_score),
            "abs_error": round(abs(pred_score - sample.true_score), 2),
        })

    return ModelResult(
        name=model_name,
        model_path=model_path,
        metrics=compute_metrics(predictions, [s.true_score for s in samples]),
        predictions=prediction_records,
    )


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    model_configs: list[dict[str, str]],
    pairs_path: Path,
    resumes_path: Path,
    jds_path: Path,
) -> dict[str, Any]:
    print("Loading test split…")
    samples = load_test_pairs(pairs_path, resumes_path, jds_path)
    print(f"  {len(samples)} test pairs loaded\n")

    results: list[ModelResult] = []
    for cfg in model_configs:
        print(f"Evaluating: {cfg['name']}")
        result = run_evaluation(cfg["name"], cfg.get("path", cfg["name"]), samples)
        results.append(result)
        m = result.metrics
        print(f"  MAE={m.mae:.4f}  RMSE={m.rmse:.4f}  LabelAcc={m.label_accuracy:.4f}\n")

    _print_comparison_table(results)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_version": "v0.2",
        "test_set_size": len(samples),
        "models": [
            {
                "name": r.name,
                "model_path": r.model_path,
                "metrics": {
                    "mae": r.metrics.mae,
                    "rmse": r.metrics.rmse,
                    "label_accuracy": r.metrics.label_accuracy,
                    "n_samples": r.metrics.n_samples,
                },
                "predictions": r.predictions,
            }
            for r in results
        ],
    }


def _print_comparison_table(results: list[ModelResult]) -> None:
    if not results:
        return
    base_mae = results[0].metrics.mae
    header = f"{'Model':<32} {'MAE':>8} {'RMSE':>8} {'LabelAcc':>10} {'ΔMAE':>10}"
    separator = "─" * len(header)
    print(separator)
    print(header)
    print(separator)
    for i, r in enumerate(results):
        delta = "baseline" if i == 0 else f"{r.metrics.mae - base_mae:+.4f}"
        print(
            f"{r.name:<32} {r.metrics.mae:>8.4f} {r.metrics.rmse:>8.4f}"
            f" {r.metrics.label_accuracy:>10.4f} {delta:>10}"
        )
    print(separator + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark similarity model versions on the test split"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
        help=(
            "Model specs as name:path. "
            "Use just name for HuggingFace models. "
            "Example: base:sentence-transformers/all-MiniLM-L6-v2 "
            "v0.2-ep3:./models/fine-tuned-miniLM-v0.2-ep3"
        ),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("datasets/versions/v0.2/cv_jd_pairs.jsonl"),
    )
    parser.add_argument(
        "--resumes",
        type=Path,
        default=Path("datasets/versions/v0.2/resumes.jsonl"),
    )
    parser.add_argument(
        "--jds",
        type=Path,
        default=Path("datasets/versions/v0.2/job_descriptions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save JSON report to this path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_configs: list[dict[str, str]] = []
    for spec in args.models:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name, path = spec, spec
        model_configs.append({"name": name, "path": path})

    report = run_pipeline(
        model_configs=model_configs,
        pairs_path=args.pairs,
        resumes_path=args.resumes,
        jds_path=args.jds,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Report saved → {args.output}")


if __name__ == "__main__":
    main()
