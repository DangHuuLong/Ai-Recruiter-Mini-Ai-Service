"""
Prepare JD section-labeling data for the JD section-classifier distillation
model. Mirrors training/prepare_section_classifier_data.py (the CV version).

Reads datasets/jd_section_splitting/vX/labeled_lines.jsonl (one JSON object
per JD, each with a `lines` array of {"text","label"}) and flattens every
JD's lines into individual training rows, split 70/15/15 into
train/validation/test at the JD level (never split lines from the same JD
across sets).

Usage:
  python -m training.prepare_jd_section_classifier_data
  python -m training.prepare_jd_section_classifier_data \\
      --input datasets/jd_section_splitting/v0.1/labeled_lines.jsonl \\
      --output datasets/jd_section_splitting/v0.1/classifier
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_INPUT = Path("datasets") / "jd_section_splitting" / "v0.1" / "labeled_lines.jsonl"
DEFAULT_OUTPUT = Path("datasets") / "jd_section_splitting" / "v0.1" / "classifier"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
# test gets the remainder (0.15)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def assign_split(doc_id: str) -> str:
    """Deterministic 70/15/15 split keyed by a stable hash of doc_id — same
    approach as prepare_section_classifier_data.py's assign_split."""
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < TRAIN_RATIO:
        return "train"
    if bucket < TRAIN_RATIO + VALIDATION_RATIO:
        return "validation"
    return "test"


def prepare(input_path: Path, output_dir: Path) -> None:
    docs = _load_jsonl(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        split: (output_dir / f"jd_section_classifier_{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }

    counts = {"train": 0, "validation": 0, "test": 0}
    line_counts = {"train": 0, "validation": 0, "test": 0}
    skipped = 0

    for doc in docs:
        doc_id = doc.get("id")
        lines = doc.get("lines")
        if not doc_id or not isinstance(lines, list) or not lines:
            skipped += 1
            continue

        split = assign_split(doc_id)
        counts[split] += 1

        for line_index, item in enumerate(lines):
            if not isinstance(item, dict) or "text" not in item or "label" not in item:
                continue
            row = {
                "doc_id": doc_id,
                "line_index": line_index,
                "text": item["text"],
                "label": item["label"],
            }
            writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            line_counts[split] += 1

    for f in writers.values():
        f.close()

    print("JD section-classifier data written to:", output_dir)
    for split in ("train", "validation", "test"):
        path = output_dir / f"jd_section_classifier_{split}.jsonl"
        print(f"  {split:<10}: {counts[split]:>5} JDs, {line_counts[split]:>6} lines  ->  {path}")
    if skipped:
        print(f"  skipped: {skipped} JD(s) (missing id or empty/malformed lines)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare JD section-classifier training data from the labeled-lines dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(input_path=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
