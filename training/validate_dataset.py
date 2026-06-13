"""Validate CV-JD dataset JSONL files.

This script is intentionally lightweight and dependency-free so it can run
before the training pipeline exists.

Usage:
    python training/validate_dataset.py
    python training/validate_dataset.py --dataset-root datasets
    python training/validate_dataset.py --version v0.1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JD_REQUIRED_FIELDS = {
    "id",
    "source",
    "title",
    "level",
    "raw_text",
    "responsibilities",
    "requirements",
    "required_skills",
    "preferred_skills",
    "min_experience_years",
    "language",
}

RESUME_REQUIRED_FIELDS = {
    "id",
    "source",
    "candidate_level",
    "raw_text",
    "skills",
    "normalized_skills",
    "experience_years",
    "education",
    "projects",
    "anonymized",
    "language",
}

PAIR_REQUIRED_FIELDS = {
    "id",
    "resume_id",
    "job_description_id",
    "split",
    "label",
    "overall_score",
    "criterion_scores",
    "matched_skills",
    "missing_required_skills",
    "label_notes",
    "label_version",
}

LEGACY_RUBRIC_KEYS = {
    "skill_match",
    "experience_match",
    "education_match",
    "domain_relevance",
    "nice_to_have",
}

SCORER_ALIGNED_RUBRIC_KEYS = {
    "SKILLS_MATCH",
    "EXPERIENCE_RELEVANCE",
    "PROJECT_RELEVANCE",
    "EDUCATION_CERTIFICATION",
    "KEYWORD_DOMAIN_ALIGNMENT",
}

RUBRIC_KEYS_BY_VERSION = {
    "rubric_v0.1": LEGACY_RUBRIC_KEYS,
    "rubric_v0.2": SCORER_ALIGNED_RUBRIC_KEYS,
}

ALLOWED_LEVELS = {"intern", "junior", "middle", "senior", "unknown"}
ALLOWED_LANGUAGES = {"en", "vi", "mixed"}
ALLOWED_SPLITS = {"train", "validation", "test", None}
ALLOWED_LABELS = {
    "excellent_match",
    "strong_match",
    "moderate_match",
    "weak_match",
    "poor_match",
}

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\s.-]?){8,15}")
PROFILE_URL_PATTERN = re.compile(r"https?://(?:www\.)?(linkedin|github)\.com/[^\s]+", re.IGNORECASE)


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

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def validate(self, dataset_root: Path, version: str | None = None) -> int:
        if version:
            base_path = dataset_root / "versions" / version
            jd_path = base_path / "job_descriptions.jsonl"
            resume_path = base_path / "resumes.jsonl"
            pair_path = base_path / "cv_jd_pairs.jsonl"
        else:
            jd_path = dataset_root / "raw" / "job_descriptions.jsonl"
            resume_path = dataset_root / "raw" / "resumes.jsonl"
            pair_path = dataset_root / "processed" / "cv_jd_pairs.jsonl"

        jd_records = self.read_jsonl(jd_path)
        resume_records = self.read_jsonl(resume_path)
        pair_records = self.read_jsonl(pair_path)

        jd_ids = self.validate_collection_ids(jd_records, "job description")
        resume_ids = self.validate_collection_ids(resume_records, "resume")
        pair_ids = self.validate_collection_ids(pair_records, "CV-JD pair")

        for record in jd_records:
            self.validate_job_description(record)

        for record in resume_records:
            self.validate_resume(record)

        for record in pair_records:
            self.validate_pair(record, resume_ids=resume_ids, jd_ids=jd_ids)

        if not jd_records:
            self.error(f"No job descriptions found in {jd_path}")
        if not resume_records:
            self.error(f"No resumes found in {resume_path}")
        if not pair_records:
            self.error(f"No CV-JD pairs found in {pair_path}")

        _ = pair_ids
        return 0 if not self.errors else 1

    def read_jsonl(self, path: Path) -> list[JsonlRecord]:
        records: list[JsonlRecord] = []

        if not path.exists():
            self.error(f"Missing file: {path}")
            return records

        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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

    def validate_collection_ids(self, records: list[JsonlRecord], label: str) -> set[str]:
        seen: set[str] = set()
        duplicate_ids: set[str] = set()

        for record in records:
            record_id = record.data.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                self.error(f"{self.location(record)} {label} has missing or invalid id")
                continue

            if record_id in seen:
                duplicate_ids.add(record_id)
            seen.add(record_id)

        for duplicate_id in sorted(duplicate_ids):
            self.error(f"Duplicate {label} id found: {duplicate_id}")

        return seen

    def validate_job_description(self, record: JsonlRecord) -> None:
        data = record.data
        self.require_fields(record, JD_REQUIRED_FIELDS)

        self.require_string(record, "id")
        self.require_string(record, "source")
        self.require_string(record, "title")
        self.require_string(record, "raw_text")
        self.require_list(record, "responsibilities")
        self.require_list(record, "requirements")
        self.require_list(record, "required_skills")
        self.require_list(record, "preferred_skills")

        if data.get("level") not in ALLOWED_LEVELS:
            self.error(f"{self.location(record)} level must be one of {sorted(ALLOWED_LEVELS)}")

        if data.get("language") not in ALLOWED_LANGUAGES:
            self.error(f"{self.location(record)} language must be one of {sorted(ALLOWED_LANGUAGES)}")

        min_experience_years = data.get("min_experience_years")
        if min_experience_years is not None and not isinstance(min_experience_years, (int, float)):
            self.error(f"{self.location(record)} min_experience_years must be a number or null")

    def validate_resume(self, record: JsonlRecord) -> None:
        data = record.data
        self.require_fields(record, RESUME_REQUIRED_FIELDS)

        self.require_string(record, "id")
        self.require_string(record, "source")
        self.require_string(record, "raw_text")
        self.require_list(record, "skills")
        self.require_list(record, "normalized_skills")
        self.require_list(record, "projects")

        if data.get("candidate_level") not in ALLOWED_LEVELS:
            self.error(f"{self.location(record)} candidate_level must be one of {sorted(ALLOWED_LEVELS)}")

        if data.get("language") not in ALLOWED_LANGUAGES:
            self.error(f"{self.location(record)} language must be one of {sorted(ALLOWED_LANGUAGES)}")

        experience_years = data.get("experience_years")
        if experience_years is not None and not isinstance(experience_years, (int, float)):
            self.error(f"{self.location(record)} experience_years must be a number or null")

        anonymized = data.get("anonymized")
        if anonymized is not True:
            self.error(f"{self.location(record)} anonymized must be true before committing resume data")

        self.check_resume_privacy(record)

    def validate_pair(self, record: JsonlRecord, resume_ids: set[str], jd_ids: set[str]) -> None:
        data = record.data
        self.require_fields(record, PAIR_REQUIRED_FIELDS)

        self.require_string(record, "id")
        self.require_string(record, "resume_id")
        self.require_string(record, "job_description_id")
        self.require_string(record, "label")
        self.require_string(record, "label_notes")
        self.require_string(record, "label_version")
        self.require_list(record, "matched_skills")
        self.require_list(record, "missing_required_skills")

        resume_id = data.get("resume_id")
        if isinstance(resume_id, str) and resume_id not in resume_ids:
            self.error(f"{self.location(record)} resume_id does not exist: {resume_id}")

        jd_id = data.get("job_description_id")
        if isinstance(jd_id, str) and jd_id not in jd_ids:
            self.error(f"{self.location(record)} job_description_id does not exist: {jd_id}")

        split = data.get("split")
        if split not in ALLOWED_SPLITS:
            self.error(f"{self.location(record)} split must be train, validation, test, or null")

        label = data.get("label")
        if label not in ALLOWED_LABELS:
            self.error(f"{self.location(record)} label must be one of {sorted(ALLOWED_LABELS)}")
        elif isinstance(label, str):
            self.label_counts[label] += 1

        overall_score = data.get("overall_score")
        if not isinstance(overall_score, int):
            self.error(f"{self.location(record)} overall_score must be an integer from 0 to 100")
        elif not 0 <= overall_score <= 100:
            self.error(f"{self.location(record)} overall_score must be between 0 and 100")
        elif isinstance(label, str):
            expected_label = self.expected_label_for_score(overall_score)
            if label != expected_label:
                self.error(
                    f"{self.location(record)} label {label!r} does not match "
                    f"overall_score {overall_score}; expected {expected_label!r}"
                )

        criterion_scores = data.get("criterion_scores")
        if not isinstance(criterion_scores, dict):
            self.error(f"{self.location(record)} criterion_scores must be an object")
            return

        label_version = data.get("label_version")
        expected_keys = RUBRIC_KEYS_BY_VERSION.get(str(label_version))
        if expected_keys is None:
            self.error(
                f"{self.location(record)} unsupported label_version {label_version!r}; "
                f"expected one of {sorted(RUBRIC_KEYS_BY_VERSION)}"
            )
            return

        missing_keys = expected_keys - set(criterion_scores)
        extra_keys = set(criterion_scores) - expected_keys

        if missing_keys:
            self.error(f"{self.location(record)} criterion_scores missing keys: {sorted(missing_keys)}")
        if extra_keys:
            self.warning(f"{self.location(record)} criterion_scores has extra keys: {sorted(extra_keys)}")

        for key, value in criterion_scores.items():
            if not isinstance(value, int) or not 0 <= value <= 100:
                self.error(f"{self.location(record)} criterion_scores.{key} must be an integer from 0 to 100")

    def require_fields(self, record: JsonlRecord, required_fields: set[str]) -> None:
        missing = required_fields - set(record.data)
        if missing:
            self.error(f"{self.location(record)} missing required fields: {sorted(missing)}")

    def require_string(self, record: JsonlRecord, field: str) -> None:
        value = record.data.get(field)
        if not isinstance(value, str) or not value.strip():
            self.error(f"{self.location(record)} {field} must be a non-empty string")

    def require_list(self, record: JsonlRecord, field: str) -> None:
        value = record.data.get(field)
        if not isinstance(value, list):
            self.error(f"{self.location(record)} {field} must be a list")

    def check_resume_privacy(self, record: JsonlRecord) -> None:
        searchable_chunks = [str(record.data.get("raw_text", "")), str(record.data.get("summary", ""))]

        for project in record.data.get("projects", []):
            if isinstance(project, dict):
                searchable_chunks.append(str(project.get("name", "")))
                searchable_chunks.append(str(project.get("description", "")))

        text = "\n".join(searchable_chunks)

        if EMAIL_PATTERN.search(text):
            self.error(f"{self.location(record)} contains an email-like pattern")
        if PHONE_PATTERN.search(text):
            self.error(f"{self.location(record)} contains a phone-like pattern")
        if PROFILE_URL_PATTERN.search(text):
            self.error(f"{self.location(record)} contains a personal profile URL pattern")

    @staticmethod
    def expected_label_for_score(score: int) -> str:
        if score >= 90:
            return "excellent_match"
        if score >= 75:
            return "strong_match"
        if score >= 60:
            return "moderate_match"
        if score >= 40:
            return "weak_match"
        return "poor_match"

    @staticmethod
    def location(record: JsonlRecord) -> str:
        return f"{record.path}:{record.line_number}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CV-JD dataset JSONL files.")
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Dataset root directory. Defaults to datasets.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Validate a versioned snapshot, for example v0.1. Defaults to working raw/processed files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and final status.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = DatasetValidator()
    exit_code = validator.validate(Path(args.dataset_root), version=args.version)

    if validator.warnings and not args.quiet:
        print("Warnings:")
        for warning in validator.warnings:
            print(f"  - {warning}")
        print()

    if validator.label_counts and not args.quiet:
        total = sum(validator.label_counts.values())
        print("Label distribution (CV-JD pairs):")
        for label in sorted(ALLOWED_LABELS, key=lambda l: -validator.label_counts[l]):
            count = validator.label_counts[label]
            pct = count / total * 100 if total else 0
            print(f"  {label:<20} {count:>4}  ({pct:.1f}%)")
        print(f"  {'TOTAL':<20} {total:>4}")
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
