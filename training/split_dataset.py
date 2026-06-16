"""Plan or generate train/validation/test splits for CV-JD pair datasets.

The dataset schema says early datasets should keep split=null until there are
at least 50 labeled pairs. This script follows that rule by default: it reports
that the dataset is not ready instead of forcing split assignment.

Usage:
    python training/split_dataset.py --dry-run
    python training/split_dataset.py --version v0.1 --dry-run
    python training/split_dataset.py --output datasets/versions/v0.2/cv_jd_pairs.jsonl
    python training/split_dataset.py --force --output /tmp/cv_jd_pairs.split.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MIN_PAIRS = 50
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42
ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class PairRecord:
    line_number: int
    data: dict[str, Any]

    @property
    def pair_id(self) -> str:
        return str(self.data.get("id", f"line_{self.line_number}"))

    @property
    def resume_id(self) -> str:
        return str(self.data.get("resume_id", ""))

    @property
    def job_description_id(self) -> str:
        return str(self.data.get("job_description_id", ""))


@dataclass(frozen=True)
class SplitPlan:
    assignments: dict[str, str]
    pair_count_by_split: dict[str, int]
    component_count_by_split: dict[str, int]
    component_count: int


class DatasetSplitPlanner:
    def __init__(
        self,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> None:
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def read_pairs(self, path: Path) -> list[PairRecord]:
        if not path.exists():
            self.errors.append(f"Missing pair dataset file: {path}")
            return []

        pairs: list[PairRecord] = []
        seen_ids: set[str] = set()

        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                self.warnings.append(f"{path}:{line_number} is empty and was skipped")
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                self.errors.append(f"{path}:{line_number} is not valid JSON: {exc.msg}")
                continue

            if not isinstance(data, dict):
                self.errors.append(f"{path}:{line_number} must be a JSON object")
                continue

            pair_id = data.get("id")
            if not isinstance(pair_id, str) or not pair_id.strip():
                self.errors.append(f"{path}:{line_number} has missing or invalid id")
                continue

            if pair_id in seen_ids:
                self.errors.append(f"Duplicate pair id found: {pair_id}")
                continue

            seen_ids.add(pair_id)
            pairs.append(PairRecord(line_number=line_number, data=data))

        return pairs

    def create_plan(self, pairs: list[PairRecord]) -> SplitPlan:
        self.validate_ratios()
        components = self.build_leakage_safe_components(pairs)
        rng = random.Random(self.seed)
        rng.shuffle(components)

        target_counts = self.calculate_target_counts(len(pairs))
        pair_count_by_split = {split: 0 for split in ALLOWED_SPLITS}
        component_count_by_split = {split: 0 for split in ALLOWED_SPLITS}
        assignments: dict[str, str] = {}

        components.sort(key=len, reverse=True)

        for component in components:
            split = self.choose_split_for_component(component, pair_count_by_split, target_counts)
            component_count_by_split[split] += 1
            pair_count_by_split[split] += len(component)

            for pair in component:
                assignments[pair.pair_id] = split

        return SplitPlan(
            assignments=assignments,
            pair_count_by_split=pair_count_by_split,
            component_count_by_split=component_count_by_split,
            component_count=len(components),
        )

    def build_leakage_safe_components(self, pairs: list[PairRecord]) -> list[list[PairRecord]]:
        """Group pairs by connected resume/JD graph components.

        If two pairs share the same resume_id or job_description_id, they must
        stay in the same split. Treating the dataset as a bipartite graph and
        splitting by connected components avoids leakage across splits.
        """

        pair_by_id = {pair.pair_id: pair for pair in pairs}
        graph: dict[str, set[str]] = defaultdict(set)

        for pair in pairs:
            resume_node = f"resume:{pair.resume_id}"
            jd_node = f"jd:{pair.job_description_id}"
            pair_node = f"pair:{pair.pair_id}"

            graph[pair_node].add(resume_node)
            graph[pair_node].add(jd_node)
            graph[resume_node].add(pair_node)
            graph[jd_node].add(pair_node)

        visited: set[str] = set()
        components: list[list[PairRecord]] = []

        for pair in pairs:
            start = f"pair:{pair.pair_id}"
            if start in visited:
                continue

            stack = [start]
            pair_ids_in_component: set[str] = set()

            while stack:
                node = stack.pop()
                if node in visited:
                    continue

                visited.add(node)
                if node.startswith("pair:"):
                    pair_ids_in_component.add(node.removeprefix("pair:"))

                stack.extend(graph[node] - visited)

            components.append([pair_by_id[pair_id] for pair_id in sorted(pair_ids_in_component)])

        return components

    def calculate_target_counts(self, pair_count: int) -> dict[str, int]:
        train_target = round(pair_count * self.train_ratio)
        validation_target = round(pair_count * self.validation_ratio)
        test_target = pair_count - train_target - validation_target

        if test_target < 0:
            test_target = 0

        return {
            "train": train_target,
            "validation": validation_target,
            "test": test_target,
        }

    def choose_split_for_component(
        self,
        component: list[PairRecord],
        current_counts: dict[str, int],
        target_counts: dict[str, int],
    ) -> str:
        remaining_capacity = {
            split: target_counts[split] - current_counts[split]
            for split in ALLOWED_SPLITS
        }

        split_order = ["train", "validation", "test"]
        return max(
            split_order,
            key=lambda split: (remaining_capacity[split], -current_counts[split]),
        )

    def validate_ratios(self) -> None:
        ratio_sum = self.train_ratio + self.validation_ratio + self.test_ratio
        if not 0.999 <= ratio_sum <= 1.001:
            self.errors.append(
                "Split ratios must sum to 1.0; "
                f"got {ratio_sum:.3f} from train={self.train_ratio}, "
                f"validation={self.validation_ratio}, test={self.test_ratio}"
            )

    def write_pairs_with_split(self, pairs: list[PairRecord], assignments: dict[str, str], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for pair in pairs:
            updated = dict(pair.data)
            updated["split"] = assignments[pair.pair_id]
            lines.append(json.dumps(updated, ensure_ascii=False, separators=(",", ":")))

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_pair_path(dataset_root: Path, version: str | None) -> Path:
    if version:
        return dataset_root / "versions" / version / "cv_jd_pairs.jsonl"
    return dataset_root / "processed" / "cv_jd_pairs.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or generate CV-JD train/validation/test splits.")
    parser.add_argument("--dataset-root", default="datasets", help="Dataset root directory. Defaults to datasets.")
    parser.add_argument("--version", default=None, help="Read pairs from a versioned snapshot, for example v0.1.")
    parser.add_argument("--output", default=None, help="Optional output JSONL path. If omitted, no file is written.")
    parser.add_argument("--dry-run", action="store_true", help="Print the split plan without writing output.")
    parser.add_argument("--force", action="store_true", help="Allow split generation below the minimum pair threshold.")
    parser.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS, help="Minimum pair count before split assignment. Defaults to 50.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for component ordering. Defaults to 42.")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train split ratio. Defaults to 0.70.")
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO, help="Validation split ratio. Defaults to 0.15.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test split ratio. Defaults to 0.15.")
    return parser.parse_args()


def print_plan(pair_path: Path, pairs: list[PairRecord], plan: SplitPlan, min_pairs: int) -> None:
    print(f"Pair dataset: {pair_path}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Leakage-safe components: {plan.component_count}")
    print(f"Minimum pairs before stable split: {min_pairs}")
    print()
    print("Planned pair counts:")
    for split in ["train", "validation", "test"]:
        print(f"  - {split}: {plan.pair_count_by_split[split]} pairs, {plan.component_count_by_split[split]} components")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    pair_path = resolve_pair_path(dataset_root, args.version)

    planner = DatasetSplitPlanner(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    pairs = planner.read_pairs(pair_path)
    if not planner.errors:
        plan = planner.create_plan(pairs)
    else:
        plan = None

    if planner.warnings:
        print("Warnings:")
        for warning in planner.warnings:
            print(f"  - {warning}")
        print()

    if planner.errors:
        print("Dataset split planning failed:")
        for error in planner.errors:
            print(f"  - {error}")
        return 1

    assert plan is not None
    print_plan(pair_path, pairs, plan, min_pairs=args.min_pairs)

    if len(pairs) < args.min_pairs and not args.force:
        print()
        print(
            f"Split generation skipped: dataset has {len(pairs)} labeled pairs, "
            f"but at least {args.min_pairs} are required before assigning stable splits."
        )
        print("Use --force only for local experiments, not for committed dataset versions.")
        return 0

    if args.dry_run or not args.output:
        print()
        print("Dry run complete. No output file was written.")
        if not args.output:
            print("Pass --output <path> to write a split JSONL file.")
        return 0

    output_path = Path(args.output)
    planner.write_pairs_with_split(pairs, plan.assignments, output_path)
    print()
    print(f"Split dataset written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
