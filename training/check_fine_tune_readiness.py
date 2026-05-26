"""Check whether the CV-JD dataset is ready for fine-tuning.

This script does not train a model. It protects the project from starting
fine-tuning too early when the dataset is still exploratory.

Usage:
    python training/check_fine_tune_readiness.py
    python training/check_fine_tune_readiness.py --version v0.1
    python training/check_fine_tune_readiness.py --min-pairs 100 --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = "datasets"
DEFAULT_MIN_PAIRS = 100
REQUIRED_SPLITS = {"train", "validation", "test"}
RECOMMENDED_LABEL_VERSION = "rubric_v0.2"


@dataclass(frozen=True)
class PairRecord:
    id: str
    resume_id: str
    job_description_id: str
    split: str | None
    label: str
    label_version: str
    overall_score: int


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


def resolve_pair_path(dataset_root: Path, version: str | None) -> Path:
    if version:
        return dataset_root / "versions" / version / "cv_jd_pairs.jsonl"
    return dataset_root / "processed" / "cv_jd_pairs.jsonl"


def parse_pairs(records: list[dict[str, Any]]) -> list[PairRecord]:
    pairs: list[PairRecord] = []

    for index, record in enumerate(records, start=1):
        pair_id = record.get("id")
        resume_id = record.get("resume_id")
        job_description_id = record.get("job_description_id")
        split = record.get("split")
        label = record.get("label")
        label_version = record.get("label_version")
        overall_score = record.get("overall_score")

        if not isinstance(pair_id, str):
            raise ValueError(f"Pair line {index} has invalid id")
        if not isinstance(resume_id, str):
            raise ValueError(f"Pair {pair_id} has invalid resume_id")
        if not isinstance(job_description_id, str):
            raise ValueError(f"Pair {pair_id} has invalid job_description_id")
        if split is not None and not isinstance(split, str):
            raise ValueError(f"Pair {pair_id} has invalid split")
        if not isinstance(label, str):
            raise ValueError(f"Pair {pair_id} has invalid label")
        if not isinstance(label_version, str):
            raise ValueError(f"Pair {pair_id} has invalid label_version")
        if not isinstance(overall_score, int):
            raise ValueError(f"Pair {pair_id} has invalid overall_score")

        pairs.append(
            PairRecord(
                id=pair_id,
                resume_id=resume_id,
                job_description_id=job_description_id,
                split=split,
                label=label,
                label_version=label_version,
                overall_score=overall_score,
            )
        )

    return pairs


def collect_entity_splits(pairs: list[PairRecord], entity: str) -> dict[str, set[str]]:
    entity_splits: dict[str, set[str]] = defaultdict(set)

    for pair in pairs:
        if pair.split is None:
            continue

        entity_id = pair.resume_id if entity == "resume" else pair.job_description_id
        entity_splits[entity_id].add(pair.split)

    return entity_splits


def check_readiness(pairs: list[PairRecord], min_pairs: int) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []

    if len(pairs) < min_pairs:
        blockers.append(f"Only {len(pairs)} labeled pairs found; at least {min_pairs} are recommended before fine-tuning.")

    label_versions = Counter(pair.label_version for pair in pairs)
    if set(label_versions) != {RECOMMENDED_LABEL_VERSION}:
        blockers.append(
            f"Dataset should use only {RECOMMENDED_LABEL_VERSION} before fine-tuning; found {dict(label_versions)}."
        )

    split_counts = Counter(pair.split for pair in pairs)
    if None in split_counts:
        blockers.append(f"{split_counts[None]} pairs still have split=null; stable train/validation/test splits are required.")

    present_splits = {split for split in split_counts if isinstance(split, str)}
    missing_splits = REQUIRED_SPLITS - present_splits
    if missing_splits:
        blockers.append(f"Missing required split(s): {sorted(missing_splits)}.")

    label_counts = Counter(pair.label for pair in pairs)
    if len(label_counts) < 3:
        warnings.append(f"Only {len(label_counts)} label class(es) found: {dict(label_counts)}. More label diversity is recommended.")

    for split in sorted(REQUIRED_SPLITS):
        split_pairs = [pair for pair in pairs if pair.split == split]
        if not split_pairs:
            continue

        split_label_counts = Counter(pair.label for pair in split_pairs)
        if len(split_label_counts) < 2:
            warnings.append(f"Split {split!r} has low label diversity: {dict(split_label_counts)}.")

    resume_splits = collect_entity_splits(pairs, "resume")
    leaked_resumes = {resume_id: splits for resume_id, splits in resume_splits.items() if len(splits) > 1}
    if leaked_resumes:
        blockers.append(f"Resume leakage detected across splits: {leaked_resumes}.")

    jd_splits = collect_entity_splits(pairs, "job_description")
    leaked_jds = {jd_id: splits for jd_id, splits in jd_splits.items() if len(splits) > 1}
    if leaked_jds:
        blockers.append(f"Job description leakage detected across splits: {leaked_jds}.")

    return blockers, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the CV-JD dataset is ready for fine-tuning.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Dataset root directory. Defaults to datasets.")
    parser.add_argument("--version", default=None, help="Check a versioned snapshot, for example v0.1.")
    parser.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS, help="Minimum labeled pair count before fine-tuning. Defaults to 100.")
    parser.add_argument("--strict", action="store_true", help="Return exit code 1 when readiness blockers exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pair_path = resolve_pair_path(Path(args.dataset_root), args.version)

    try:
        records = read_jsonl(pair_path)
        pairs = parse_pairs(records)
        blockers, warnings = check_readiness(pairs, min_pairs=args.min_pairs)
    except Exception as exc:
        print(f"Fine-tune readiness check failed: {exc}", file=sys.stderr)
        return 1

    print(f"Pair dataset: {pair_path}")
    print(f"Total pairs: {len(pairs)}")
    print()

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    if blockers:
        print("Fine-tuning is NOT ready yet.")
        print("Blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
        print()
        print("Next action: collect more labeled CV-JD pairs, assign stable splits, then re-run this check.")
        return 1 if args.strict else 0

    print("Fine-tuning readiness check passed.")
    print("Next action: create a separate fine-tuning experiment branch and keep generated artifacts out of git.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
