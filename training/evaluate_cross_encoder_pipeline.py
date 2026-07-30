"""
Evaluate 2-stage cross-encoder pipeline on the v0.3 test split.

  Stage 1 — bi-encoder (v0.3): encode CV and JD separately, retrieve top-K CVs per JD
  Stage 2 — cross-encoder: rerank top-K, produce calibrated score

Metrics reported
  recall@K        : fraction of test pairs where CV is in the bi-encoder top-K for its JD
  MAE/RMSE/LabelAcc: for (a) bi-encoder only, (b) cross-encoder only, (c) 2-stage pipeline

Cross-encoder calibration: sigmoid(logit) × 100 → 0–100 range.
MAE/RMSE for the cross-encoder approaches may differ from the bi-encoder due to this
uncalibrated mapping; focus on recall@K and label accuracy for a fair comparison.

Usage:
  python -m training.evaluate_cross_encoder_pipeline
  python -m training.evaluate_cross_encoder_pipeline \\
      --biencoder models/fine-tuned-miniLM-v0.3 \\
      --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \\
      --top-k 50
  python -m training.evaluate_cross_encoder_pipeline --skip-cross-encoder
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from training.evaluate_similarity_pipeline import (
    _extract_jd_text,
    _extract_resume_text,
    _load_jsonl,
    compute_metrics,
    label_for_score,
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    pair_id: str
    resume_id: str
    jd_id: str
    true_score: float
    true_label: str
    cv_text: str
    jd_text: str


def load_samples(
    pairs_path: Path,
    resumes_path: Path,
    jds_path: Path,
    split: str = "test",
) -> list[Sample]:
    resumes = {r["id"]: r for r in _load_jsonl(resumes_path)}
    jds = {j["id"]: j for j in _load_jsonl(jds_path)}
    samples: list[Sample] = []
    for pair in _load_jsonl(pairs_path):
        if split and pair.get("split") != split:
            continue
        resume = resumes.get(pair["resume_id"])
        jd = jds.get(pair["job_description_id"])
        if resume is None or jd is None:
            continue
        true_score = float(pair["overall_score"])
        samples.append(Sample(
            pair_id=pair["id"],
            resume_id=pair["resume_id"],
            jd_id=pair["job_description_id"],
            true_score=true_score,
            true_label=pair.get("label") or label_for_score(true_score),
            cv_text=_extract_resume_text(resume),
            jd_text=_extract_jd_text(jd),
        ))
    return samples


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def _load_calibrator(path: str | None) -> Any:
    """Load a fitted isotonic calibrator from path, or return None if not found."""
    import pickle

    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("rb") as f:
        cal = pickle.load(f)
    print(f"  [calibrator loaded from {p}]")
    return cal


# ── Stage 1: Bi-encoder ───────────────────────────────────────────────────────

def run_biencoder(
    model_path: str,
    samples: list[Sample],
    batch_size: int = 64,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Encode unique CVs and JDs.
    Returns (pair_id → score, resume_id → embedding, jd_id → embedding).
    """
    from sentence_transformers import SentenceTransformer

    print(f"  [bi-encoder] Loading: {model_path}")
    model = SentenceTransformer(model_path)

    unique_cv: dict[str, str] = {}
    unique_jd: dict[str, str] = {}
    for s in samples:
        unique_cv[s.resume_id] = s.cv_text
        unique_jd[s.jd_id] = s.jd_text

    cv_ids = list(unique_cv.keys())
    jd_ids = list(unique_jd.keys())

    print(f"  [bi-encoder] Encoding {len(cv_ids)} CVs + {len(jd_ids)} JDs…")
    all_embs = model.encode(
        [unique_cv[i] for i in cv_ids] + [unique_jd[i] for i in jd_ids],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    cv_embs: dict[str, np.ndarray] = {cv_ids[i]: all_embs[i] for i in range(len(cv_ids))}
    jd_embs: dict[str, np.ndarray] = {
        jd_ids[i]: all_embs[len(cv_ids) + i] for i in range(len(jd_ids))
    }

    scores: dict[str, float] = {}
    for s in samples:
        sim = float(np.dot(cv_embs[s.resume_id], jd_embs[s.jd_id]))
        scores[s.pair_id] = round(max(0.0, min(1.0, sim)) * 100, 2)

    return scores, cv_embs, jd_embs


# ── Retrieval: top-K per JD ───────────────────────────────────────────────────

def build_top_k(
    samples: list[Sample],
    cv_embs: dict[str, np.ndarray],
    jd_embs: dict[str, np.ndarray],
    k: int,
) -> dict[str, list[str]]:
    """For each unique JD in samples, return the top-K resume_ids by cosine similarity."""
    all_cv_ids = list(cv_embs.keys())
    cv_matrix = np.stack([cv_embs[cid] for cid in all_cv_ids])  # (N_cv, D)

    top_k_map: dict[str, list[str]] = {}
    for jd_id in set(s.jd_id for s in samples):
        jd_vec = jd_embs[jd_id]  # (D,)
        sims = cv_matrix @ jd_vec  # (N_cv,)
        top_indices = np.argsort(sims)[::-1][:k]
        top_k_map[jd_id] = [all_cv_ids[i] for i in top_indices]

    return top_k_map


def compute_recall_at_k(
    samples: list[Sample],
    top_k_map: dict[str, list[str]],
    k: int,
    score_thresholds: tuple[float, ...] = (0.0, 60.0, 75.0),
) -> dict[str, Any]:
    """
    recall@K = fraction of test pairs where the true CV is in top-K for its JD.
    Reports overall recall plus stratified recall per score_threshold bucket:
      threshold=0   → all pairs (baseline)
      threshold=60  → moderate+ match pairs (score ≥ 60)
      threshold=75  → strong+ match pairs (score ≥ 75)
    """
    jd_to_pairs: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        jd_to_pairs[s.jd_id].append(s)

    stratified: dict[float, dict[str, int]] = {t: {"hits": 0, "total": 0} for t in score_thresholds}

    for jd_id, jd_pairs in jd_to_pairs.items():
        top_k_set = set(top_k_map[jd_id])
        for s in jd_pairs:
            in_topk = s.resume_id in top_k_set
            for t in score_thresholds:
                if s.true_score >= t:
                    stratified[t]["total"] += 1
                    if in_topk:
                        stratified[t]["hits"] += 1

    unique_jds = len(jd_to_pairs)
    unique_cvs = len({s.resume_id for s in samples})

    result: dict[str, Any] = {
        "k": k,
        "unique_jds": unique_jds,
        "unique_cvs": unique_cvs,
    }
    for t, counts in stratified.items():
        total = counts["total"]
        hits = counts["hits"]
        key = "recall_at_k" if t == 0.0 else f"recall_at_k_score_ge_{int(t)}"
        result[key] = round(hits / total, 4) if total else 0.0
        result[f"_detail_{int(t)}"] = {"hits": hits, "total": total}

    return result


# ── Stage 2: Cross-encoder ────────────────────────────────────────────────────

def run_crossencoder(
    model_path: str,
    samples: list[Sample],
    max_length: int = 512,
    batch_size: int = 32,
) -> dict[str, float]:
    """Score all pairs. Returns pair_id → sigmoid(logit) × 100."""
    from sentence_transformers import CrossEncoder

    print(f"  [cross-encoder] Loading: {model_path}")
    model = CrossEncoder(model_path, max_length=max_length)

    print(f"  [cross-encoder] Scoring {len(samples)} pairs…")
    logits = model.predict(
        [(s.cv_text, s.jd_text) for s in samples],
        batch_size=batch_size,
        show_progress_bar=True,
    )

    return {s.pair_id: round(_sigmoid(float(logit)) * 100, 2) for s, logit in zip(samples, logits)}


# ── 2-stage pipeline ──────────────────────────────────────────────────────────

def build_pipeline_scores(
    samples: list[Sample],
    biencoder_scores: dict[str, float],
    crossencoder_scores: dict[str, float],
    top_k_map: dict[str, list[str]],
) -> tuple[dict[str, float], int]:
    """
    Use cross-encoder score for pairs where the CV was in the bi-encoder top-K.
    Fall back to bi-encoder score for pairs outside top-K.
    Returns (pair_id → final score, n_reranked).
    """
    final_scores: dict[str, float] = {}
    n_reranked = 0
    for s in samples:
        top_k_set = set(top_k_map[s.jd_id])
        if s.resume_id in top_k_set:
            final_scores[s.pair_id] = crossencoder_scores[s.pair_id]
            n_reranked += 1
        else:
            final_scores[s.pair_id] = biencoder_scores[s.pair_id]
    return final_scores, n_reranked


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_metrics_table(rows: list[tuple[str, Any]]) -> None:
    base_mae = rows[0][1].mae
    header = f"{'Approach':<38} {'MAE':>8} {'RMSE':>8} {'LabelAcc':>10} {'ΔMAE':>10}"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i, (name, m) in enumerate(rows):
        delta = "baseline" if i == 0 else f"{m.mae - base_mae:+.4f}"
        print(
            f"{name:<38} {m.mae:>8.4f} {m.rmse:>8.4f}"
            f" {m.label_accuracy:>10.4f} {delta:>10}"
        )
    print(sep + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 2-stage cross-encoder pipeline on v0.3 test split"
    )
    parser.add_argument(
        "--biencoder",
        default="models/fine-tuned-miniLM-v0.3",
    )
    parser.add_argument(
        "--cross-encoder",
        default="models/cross-encoder-cv-jd-v0.1",
        dest="cross_encoder",
    )
    parser.add_argument("--top-k", type=int, default=50, dest="top_k")
    parser.add_argument("--max-length", type=int, default=512, dest="max_length")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("datasets/versions/v0.3/cv_jd_pairs.jsonl"),
    )
    parser.add_argument(
        "--resumes",
        type=Path,
        default=Path("datasets/versions/v0.3/resumes.jsonl"),
    )
    parser.add_argument(
        "--jds",
        type=Path,
        default=Path("datasets/versions/v0.3/job_descriptions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/cross_encoder_pipeline_v0.1.json"),
    )
    parser.add_argument(
        "--skip-cross-encoder",
        action="store_true",
        dest="skip_ce",
        help="Only run bi-encoder + recall@K (skips cross-encoder, fast)",
    )
    parser.add_argument(
        "--calibrator",
        default=None,
        dest="calibrator",
        help=(
            "Path to calibrator.pkl. "
            "If omitted, auto-detects <cross_encoder_dir>/calibrator.pkl. "
            "Pass 'none' to disable."
        ),
    )
    args = parser.parse_args()

    print("Loading test split…")
    samples = load_samples(args.pairs, args.resumes, args.jds, split="test")
    print(f"  {len(samples)} test pairs loaded\n")

    true_scores = [s.true_score for s in samples]

    # ── Stage 1: Bi-encoder ─────────────────────────────────────────────────
    print("── Stage 1: Bi-encoder ──────────────────────────────────────────")
    bi_scores, cv_embs, jd_embs = run_biencoder(
        args.biencoder, samples, batch_size=args.batch_size * 2
    )
    bi_preds = [bi_scores[s.pair_id] for s in samples]
    bi_metrics = compute_metrics(bi_preds, true_scores)
    print(
        f"  MAE={bi_metrics.mae:.4f}  RMSE={bi_metrics.rmse:.4f}"
        f"  LabelAcc={bi_metrics.label_accuracy:.4f}\n"
    )

    # ── Recall@K ─────────────────────────────────────────────────────────────
    print(f"── Recall@{args.top_k} (bi-encoder retrieval) ──────────────────────")
    top_k_map = build_top_k(samples, cv_embs, jd_embs, k=args.top_k)
    recall_result = compute_recall_at_k(samples, top_k_map, k=args.top_k)
    d0  = recall_result["_detail_0"]
    d60 = recall_result["_detail_60"]
    d75 = recall_result["_detail_75"]
    print(f"  recall@{args.top_k} (all pairs)      = {recall_result['recall_at_k']:.4f}"
          f"  ({d0['hits']}/{d0['total']})")
    print(f"  recall@{args.top_k} (score ≥ 60)     = {recall_result['recall_at_k_score_ge_60']:.4f}"
          f"  ({d60['hits']}/{d60['total']})")
    print(f"  recall@{args.top_k} (score ≥ 75)     = {recall_result['recall_at_k_score_ge_75']:.4f}"
          f"  ({d75['hits']}/{d75['total']})")
    print(f"  unique JDs = {recall_result['unique_jds']}"
          f"  |  unique CVs = {recall_result['unique_cvs']}\n")

    if args.skip_ce:
        print("Skipping cross-encoder (--skip-cross-encoder)\n")
        _print_metrics_table([("bi-encoder (v0.3)", bi_metrics)])
        return

    # ── Stage 2: Cross-encoder on all test pairs ─────────────────────────────
    print("── Cross-encoder only (all pairs) ───────────────────────────────")
    ce_scores = run_crossencoder(
        args.cross_encoder, samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    ce_preds = [ce_scores[s.pair_id] for s in samples]
    ce_metrics = compute_metrics(ce_preds, true_scores)
    print(
        f"  raw  MAE={ce_metrics.mae:.4f}  RMSE={ce_metrics.rmse:.4f}"
        f"  LabelAcc={ce_metrics.label_accuracy:.4f}\n"
    )

    # ── Optional calibration ──────────────────────────────────────────────────
    cal_path = (
        None if args.calibrator == "none"
        else args.calibrator or str(Path(args.cross_encoder) / "calibrator.pkl")
    )
    calibrator = _load_calibrator(cal_path)
    if calibrator is not None:
        cal_raw = np.array([ce_scores[s.pair_id] for s in samples])
        cal_vals = calibrator.predict(cal_raw).tolist()
        ce_scores_cal: dict[str, float] = {s.pair_id: float(v) for s, v in zip(samples, cal_vals)}
        ce_preds_cal = [ce_scores_cal[s.pair_id] for s in samples]
        ce_metrics_cal = compute_metrics(ce_preds_cal, true_scores)
        print(
            f"  cal  MAE={ce_metrics_cal.mae:.4f}  RMSE={ce_metrics_cal.rmse:.4f}"
            f"  LabelAcc={ce_metrics_cal.label_accuracy:.4f}\n"
        )
    else:
        ce_scores_cal = ce_scores
        ce_metrics_cal = None

    # ── 2-stage pipeline ─────────────────────────────────────────────────────
    cal_label = " cal" if ce_metrics_cal is not None else ""
    print(f"── 2-stage pipeline (bi-top-{args.top_k} → cross-encoder rerank{cal_label}) ──")
    pipeline_scores, n_reranked = build_pipeline_scores(
        samples, bi_scores, ce_scores_cal, top_k_map
    )
    pipeline_preds = [pipeline_scores[s.pair_id] for s in samples]
    pipeline_metrics = compute_metrics(pipeline_preds, true_scores)
    print(
        f"  Reranked {n_reranked}/{len(samples)} pairs  "
        f"({len(samples) - n_reranked} fallback to bi-encoder)"
    )
    print(
        f"  MAE={pipeline_metrics.mae:.4f}  RMSE={pipeline_metrics.rmse:.4f}"
        f"  LabelAcc={pipeline_metrics.label_accuracy:.4f}\n"
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("── Summary ──────────────────────────────────────────────────────")
    table_rows: list[tuple[str, Any]] = [
        ("bi-encoder only (v0.3)", bi_metrics),
        ("cross-encoder raw", ce_metrics),
    ]
    if ce_metrics_cal is not None:
        table_rows.append(("cross-encoder calibrated", ce_metrics_cal))
    table_rows.append((f"2-stage pipeline (top-{args.top_k}){cal_label}", pipeline_metrics))
    _print_metrics_table(table_rows)
    if ce_metrics_cal is None:
        print(
            f"  * cross-encoder scores: sigmoid(logit)×100, uncalibrated.\n"
            f"    Run: python -m training.calibrate_cross_encoder\n"
        )

    # ── Save report ───────────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ce_cal_entry: dict[str, Any] = {
            "mae": ce_metrics_cal.mae,
            "rmse": ce_metrics_cal.rmse,
            "label_accuracy": ce_metrics_cal.label_accuracy,
            "n_samples": ce_metrics_cal.n_samples,
            "calibration": "isotonic_regression",
        } if ce_metrics_cal is not None else {}
        report: dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config": {
                "biencoder": args.biencoder,
                "cross_encoder": args.cross_encoder,
                "calibrator": cal_path,
                "top_k": args.top_k,
                "max_length": args.max_length,
            },
            "recall": recall_result,
            "metrics": {
                "biencoder": {
                    "mae": bi_metrics.mae,
                    "rmse": bi_metrics.rmse,
                    "label_accuracy": bi_metrics.label_accuracy,
                    "n_samples": bi_metrics.n_samples,
                },
                "cross_encoder_raw": {
                    "mae": ce_metrics.mae,
                    "rmse": ce_metrics.rmse,
                    "label_accuracy": ce_metrics.label_accuracy,
                    "n_samples": ce_metrics.n_samples,
                    "calibration": "sigmoid(logit)*100",
                },
                **({"cross_encoder_calibrated": ce_cal_entry} if ce_cal_entry else {}),
                "pipeline": {
                    "mae": pipeline_metrics.mae,
                    "rmse": pipeline_metrics.rmse,
                    "label_accuracy": pipeline_metrics.label_accuracy,
                    "n_samples": pipeline_metrics.n_samples,
                    "n_reranked": n_reranked,
                    "calibration": (
                        "isotonic_regression for reranked, cosine*100 for fallback"
                        if ce_metrics_cal is not None
                        else "sigmoid(logit)*100 for reranked, cosine*100 for fallback"
                    ),
                },
            },
        }
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Report saved → {args.output}")


if __name__ == "__main__":
    main()
