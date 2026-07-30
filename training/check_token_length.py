"""
Check token length distribution of combined CV+JD text on dataset v0.3.
Used to decide truncation strategy before cross-encoder training/inference.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from training.evaluate_similarity_pipeline import _extract_jd_text, _extract_resume_text


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _percentile(sorted_data: list[float], p: float) -> float:
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _ascii_histogram(values: list[int], buckets: int = 10) -> str:
    lo, hi = min(values), max(values)
    width = (hi - lo) / buckets or 1
    counts = [0] * buckets
    for v in values:
        idx = min(int((v - lo) / width), buckets - 1)
        counts[idx] += 1
    bar_max = max(counts)
    lines = []
    for i, count in enumerate(counts):
        bucket_lo = int(lo + i * width)
        bucket_hi = int(lo + (i + 1) * width)
        bar = "█" * int(count / bar_max * 40)
        lines.append(f"  {bucket_lo:>5}–{bucket_hi:<5} │{bar:<40}│ {count}")
    return "\n".join(lines)


def run(
    pairs_path: Path,
    resumes_path: Path,
    jds_path: Path,
    tokenizer_name: str,
    sep_token: str,
) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers is not installed") from e

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    resumes = {r["id"]: r for r in _load_jsonl(resumes_path)}
    jds = {j["id"]: j for j in _load_jsonl(jds_path)}

    cv_lengths: list[int] = []
    jd_lengths: list[int] = []
    combined_lengths: list[int] = []
    over_256 = 0
    over_512 = 0

    pairs = _load_jsonl(pairs_path)
    print(f"Processing {len(pairs)} pairs…\n")

    for pair in pairs:
        resume = resumes.get(pair["resume_id"])
        jd = jds.get(pair["job_description_id"])
        if resume is None or jd is None:
            continue

        cv_text = _extract_resume_text(resume)
        jd_text = _extract_jd_text(jd)
        combined = cv_text + f" {sep_token} " + jd_text

        cv_len = len(tokenizer.encode(cv_text, add_special_tokens=False))
        jd_len = len(tokenizer.encode(jd_text, add_special_tokens=False))
        # +3 for [CLS], [SEP] between, [SEP] at end
        combined_len = len(tokenizer.encode(combined, add_special_tokens=True))

        cv_lengths.append(cv_len)
        jd_lengths.append(jd_len)
        combined_lengths.append(combined_len)

        if combined_len > 256:
            over_256 += 1
        if combined_len > 512:
            over_512 += 1

    sorted_combined = sorted(combined_lengths)
    n = len(sorted_combined)

    print("── CV text ──────────────────────────────────")
    print(f"  mean   : {statistics.mean(cv_lengths):.1f}")
    print(f"  median : {statistics.median(cv_lengths):.1f}")
    print(f"  max    : {max(cv_lengths)}")

    print("\n── JD text ──────────────────────────────────")
    print(f"  mean   : {statistics.mean(jd_lengths):.1f}")
    print(f"  median : {statistics.median(jd_lengths):.1f}")
    print(f"  max    : {max(jd_lengths)}")

    print("\n── Combined (CV [SEP] JD) ───────────────────")
    print(f"  n      : {n}")
    print(f"  min    : {min(combined_lengths)}")
    print(f"  mean   : {statistics.mean(combined_lengths):.1f}")
    print(f"  median : {statistics.median(combined_lengths):.1f}")
    print(f"  p75    : {_percentile(sorted_combined, 75):.0f}")
    print(f"  p90    : {_percentile(sorted_combined, 90):.0f}")
    print(f"  p95    : {_percentile(sorted_combined, 95):.0f}")
    print(f"  p99    : {_percentile(sorted_combined, 99):.0f}")
    print(f"  max    : {max(combined_lengths)}")
    print(f"\n  > 256 tokens : {over_256:>5} ({over_256/n*100:.1f}%)")
    print(f"  > 512 tokens : {over_512:>5} ({over_512/n*100:.1f}%)")

    print("\n── Distribution (combined tokens) ──────────")
    print(_ascii_histogram(combined_lengths))

    print("\n── Verdict ──────────────────────────────────")
    p95 = _percentile(sorted_combined, 95)
    if p95 <= 256:
        print("  ✓ p95 ≤ 256 — max_length=256 covers 95% of pairs (fast inference)")
    elif p95 <= 512:
        print("  ✓ p95 ≤ 512 — max_length=512 covers 95% of pairs (standard BERT limit)")
    else:
        print("  ✗ p95 > 512 — truncation will affect >5% of pairs, consider chunking strategy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check token length for cross-encoder input")
    parser.add_argument("--pairs", type=Path, default=Path("datasets/versions/v0.3/cv_jd_pairs.jsonl"))
    parser.add_argument("--resumes", type=Path, default=Path("datasets/versions/v0.3/resumes.jsonl"))
    parser.add_argument("--jds", type=Path, default=Path("datasets/versions/v0.3/job_descriptions.jsonl"))
    parser.add_argument(
        "--tokenizer",
        default="models/fine-tuned-miniLM-v0.3",
        help="Tokenizer to use (local path or HuggingFace model id)",
    )
    parser.add_argument("--sep-token", default="[SEP]", help="Separator token between CV and JD")
    args = parser.parse_args()

    run(
        pairs_path=args.pairs,
        resumes_path=args.resumes,
        jds_path=args.jds,
        tokenizer_name=args.tokenizer,
        sep_token=args.sep_token,
    )


if __name__ == "__main__":
    main()
