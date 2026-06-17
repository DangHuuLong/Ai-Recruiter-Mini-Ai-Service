"""
Prepare CV-JD pairs for cross-encoder fine-tuning.

Reads the raw v0.3 dataset, extracts text from CV and JD, and writes
three JSONL files (train / val / test) with the format:

  {"pair_id": "...", "cv_text": "...", "jd_text": "...",
   "score": 75.0, "label": 0.75}

  label = score / 100  (normalized to 0–1 for MSE regression loss)

These files are uploaded to Google Drive and loaded by the Colab notebook.

Usage:
  python -m training.prepare_cross_encoder_data
  python -m training.prepare_cross_encoder_data \\
      --pairs  datasets/versions/v0.3/cv_jd_pairs.jsonl \\
      --output datasets/versions/v0.3/cross_encoder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.evaluate_similarity_pipeline import (
    _extract_jd_text,
    _extract_resume_text,
    _load_jsonl,
    label_for_score,
)


def prepare(
    pairs_path: Path,
    resumes_path: Path,
    jds_path: Path,
    output_dir: Path,
) -> None:
    resumes = {r["id"]: r for r in _load_jsonl(resumes_path)}
    jds = {j["id"]: j for j in _load_jsonl(jds_path)}

    counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    writers: dict[str, object] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        writers[split] = (output_dir / f"cross_encoder_{split}.jsonl").open(
            "w", encoding="utf-8"
        )

    skipped = 0
    for pair in _load_jsonl(pairs_path):
        split = pair.get("split")
        if split not in writers:
            skipped += 1
            continue

        resume = resumes.get(pair["resume_id"])
        jd = jds.get(pair["job_description_id"])
        if resume is None or jd is None:
            skipped += 1
            continue

        score = float(pair["overall_score"])
        record = {
            "pair_id": pair["id"],
            "cv_text": _extract_resume_text(resume),
            "jd_text": _extract_jd_text(jd),
            "score": score,
            "label": round(score / 100, 6),
            "true_label": pair.get("label") or label_for_score(score),
        }
        writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
        counts[split] += 1

    for f in writers.values():
        f.close()  # type: ignore[union-attr]

    print("Cross-encoder data written to:", output_dir)
    for split, n in counts.items():
        path = output_dir / f"cross_encoder_{split}.jsonl"
        print(f"  {split:<6}: {n:>5} pairs  ->  {path}")
    if skipped:
        print(f"  skipped: {skipped} pairs (missing split field or missing CV/JD)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare cross-encoder training data from v0.3 dataset"
    )
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
        default=Path("datasets/versions/v0.3/cross_encoder"),
    )
    args = parser.parse_args()

    prepare(
        pairs_path=args.pairs,
        resumes_path=args.resumes,
        jds_path=args.jds,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
