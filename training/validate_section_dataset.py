"""Validate the section-labeling dataset (datasets/section_splitting/vX/labeled_lines.jsonl).

Mirrors training/validate_dataset.py's structure (dependency-free, dataclass
record wrapper, required-fields + allowed-value checks, label distribution
report) but for the section-classifier distillation dataset produced by
scripts/generate_section_labeling_data.py.

Usage:
    python training/validate_section_dataset.py
    python training/validate_section_dataset.py --input datasets/section_splitting/v0.1/labeled_lines.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.parsers.section_splitter import SECTION_ORDER  # noqa: E402

ALLOWED_LABELS = set(SECTION_ORDER) | {"other"}
RESUME_REQUIRED_FIELDS = {"id", "lines"}
LINE_REQUIRED_FIELDS = {"text", "label"}

MIN_LINES_PER_RESUME = 15
MIN_DISTINCT_LABELS = 4

DEFAULT_INPUT = Path("datasets") / "section_splitting" / "v0.1" / "labeled_lines.jsonl"


@dataclass(frozen=True)
class JsonlRecord:
    path: Path
    line_number: int
    data: dict[str, Any]


class DatasetValidator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.label_counts: Counter[str] = Counter()
        self.resume_count = 0
        self.line_count = 0

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def validate(self, input_path: Path) -> int:
        records = self.read_jsonl(input_path)
        resume_ids = self.validate_ids(records)

        for record in records:
            self.validate_resume(record)

        if not records:
            self.error(f"No labeled resumes found in {input_path}")

        _ = resume_ids
        return 0 if not self.errors else 1

    def read_jsonl(self, path: Path) -> list[JsonlRecord]:
        records: list[JsonlRecord] = []

        if not path.exists():
            self.error(f"Missing file: {path}")
            return records

        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                self.warning(f"{path}:{line_number} is empty and was skipped")
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                self.error(f"{path}:{line_number} is not valid JSON: {exc.msg}")
                continue

            if not isinstance(data, dict):
                self.error(f"{path}:{line_number} must be a JSON object")
                continue

            records.append(JsonlRecord(path=path, line_number=line_number, data=data))

        return records

    def validate_ids(self, records: list[JsonlRecord]) -> set[str]:
        seen: set[str] = set()
        duplicate_ids: set[str] = set()

        for record in records:
            record_id = record.data.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                self.error(f"{self.location(record)} resume has missing or invalid id")
                continue
            if record_id in seen:
                duplicate_ids.add(record_id)
            seen.add(record_id)

        for duplicate_id in sorted(duplicate_ids):
            self.error(f"Duplicate resume id found: {duplicate_id}")

        return seen

    def validate_resume(self, record: JsonlRecord) -> None:
        data = record.data
        missing = RESUME_REQUIRED_FIELDS - set(data)
        if missing:
            self.error(f"{self.location(record)} missing required fields: {sorted(missing)}")
            return

        lines = data.get("lines")
        if not isinstance(lines, list) or not lines:
            self.error(f"{self.location(record)} 'lines' must be a non-empty list")
            return

        self.resume_count += 1

        if len(lines) < MIN_LINES_PER_RESUME:
            self.warning(
                f"{self.location(record)} resume {data.get('id')} has only {len(lines)} lines "
                f"(recommended minimum {MIN_LINES_PER_RESUME})"
            )

        labels_seen: set[str] = set()
        for i, item in enumerate(lines, 1):
            if not isinstance(item, dict):
                self.error(f"{self.location(record)} resume {data.get('id')} line {i} must be an object")
                continue

            missing_line_fields = LINE_REQUIRED_FIELDS - set(item)
            if missing_line_fields:
                self.error(
                    f"{self.location(record)} resume {data.get('id')} line {i} "
                    f"missing fields: {sorted(missing_line_fields)}"
                )
                continue

            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                self.error(f"{self.location(record)} resume {data.get('id')} line {i} 'text' must be a non-empty string")

            label = item.get("label")
            if label not in ALLOWED_LABELS:
                self.error(
                    f"{self.location(record)} resume {data.get('id')} line {i} "
                    f"invalid label {label!r} (allowed: {sorted(ALLOWED_LABELS)})"
                )
                continue

            labels_seen.add(label)
            self.label_counts[label] += 1
            self.line_count += 1

        if len(labels_seen) < MIN_DISTINCT_LABELS:
            self.warning(
                f"{self.location(record)} resume {data.get('id')} uses only {len(labels_seen)} "
                f"distinct label(s) {sorted(labels_seen)} (recommended minimum {MIN_DISTINCT_LABELS})"
            )

    @staticmethod
    def location(record: JsonlRecord) -> str:
        return f"{record.path}:{record.line_number}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the section-labeling dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to labeled_lines.jsonl.")
    parser.add_argument("--quiet", action="store_true", help="Only print errors and final status.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = DatasetValidator()
    exit_code = validator.validate(Path(args.input))

    if validator.warnings and not args.quiet:
        print("Warnings:")
        for warning in validator.warnings:
            print(f"  - {warning}")
        print()

    if validator.label_counts and not args.quiet:
        total = sum(validator.label_counts.values())
        print(f"Resumes: {validator.resume_count}   Labeled lines: {validator.line_count}")
        print("Label distribution (lines):")
        for label in sorted(ALLOWED_LABELS, key=lambda l: -validator.label_counts[l]):
            count = validator.label_counts[label]
            pct = count / total * 100 if total else 0
            print(f"  {label:<16} {count:>6}  ({pct:.1f}%)")
        print()

    if validator.errors:
        print("Dataset validation failed:")
        for error in validator.errors:
            print(f"  - {error}")
        return exit_code

    print("Dataset validation passed.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
