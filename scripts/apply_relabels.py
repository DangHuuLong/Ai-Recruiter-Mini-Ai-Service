"""
Apply LLM relabeling responses back to cv_jd_pairs.jsonl.

Reads all *.json files from artifacts/relabeling/responses/ and updates
the pairs file with new labels, criterion scores, and notes.

Usage:
  python scripts/apply_relabels.py --dry-run          # preview changes only
  python scripts/apply_relabels.py                    # apply to working dataset
  python scripts/apply_relabels.py --zone weak_moderate_60  # apply one zone only
  python scripts/apply_relabels.py --pairs datasets/versions/v0.2/cv_jd_pairs.jsonl
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

RESPONSES_DIR = Path("artifacts/relabeling/responses")
PAIRS_DEFAULT = Path("datasets/processed/cv_jd_pairs.jsonl")

SCORE_RANGES = {
    "poor_match":      (0,  39),
    "weak_match":      (40, 59),
    "moderate_match":  (60, 74),
    "strong_match":    (75, 89),
    "excellent_match": (90, 100),
}

BOUNDARY_ZONES = [(35, 45), (55, 65), (70, 80), (85, 95)]

CRITERIA_KEYS = [
    "SKILLS_MATCH",
    "EXPERIENCE_RELEVANCE",
    "PROJECT_RELEVANCE",
    "EDUCATION_CERTIFICATION",
    "KEYWORD_DOMAIN_ALIGNMENT",
]

# Common LLM typos for criterion key names
CRITERIA_KEY_ALIASES = {
    "EXPERIENCE_RELEVENCE":    "EXPERIENCE_RELEVANCE",
    "EXPERIENCE_RELEVANCY":    "EXPERIENCE_RELEVANCE",
    "PROJECT_RELEVENCE":       "PROJECT_RELEVANCE",
    "PROJECT_REVELANCE":       "PROJECT_RELEVANCE",
    "PROJECT_RELEVANCY":       "PROJECT_RELEVANCE",
    "SKILS_MATCH":             "SKILLS_MATCH",
    "SKILL_MATCH":             "SKILLS_MATCH",
    "KEYWORD_ALIGNMENT":       "KEYWORD_DOMAIN_ALIGNMENT",
    "KEYWORD_DOMAIN_MATCH":    "KEYWORD_DOMAIN_ALIGNMENT",
}

CRITERIA_WEIGHTS = {
    "SKILLS_MATCH":            0.35,
    "EXPERIENCE_RELEVANCE":    0.30,
    "PROJECT_RELEVANCE":       0.15,
    "EDUCATION_CERTIFICATION": 0.10,
    "KEYWORD_DOMAIN_ALIGNMENT":0.10,
}


def score_to_label(score: int) -> str:
    for label, (lo, hi) in SCORE_RANGES.items():
        if lo <= score <= hi:
            return label
    return "unknown"


def normalize_entry(entry: dict) -> None:
    """Fix common LLM output issues in-place before validation."""
    criteria = entry.get("criterion_scores", {})
    # Rename known typo keys
    fixed = {CRITERIA_KEY_ALIASES.get(k, k): v for k, v in criteria.items()}
    # Fill missing criteria keys with 0
    for k in CRITERIA_KEYS:
        if k not in fixed:
            fixed[k] = 0
    entry["criterion_scores"] = fixed

    # Auto-correct label when score is out of its declared range (trust score)
    score = entry.get("overall_score")
    label = entry.get("label")
    if score is not None and label in SCORE_RANGES:
        lo, hi = SCORE_RANGES[label]
        if not (lo <= int(score) <= hi):
            entry["label"] = score_to_label(int(score))


def in_boundary_zone(score: int) -> bool:
    return any(lo <= score <= hi for lo, hi in BOUNDARY_ZONES)


def weighted_score(criteria: dict) -> int:
    total = sum(criteria.get(k, 0) * w for k, w in CRITERIA_WEIGHTS.items())
    return round(total)


def validate_relabel(entry: dict) -> tuple[bool, str]:
    """Return (valid, reason). Validates a single relabel entry."""
    pid = entry.get("pair_id", "?")
    label = entry.get("label")
    score = entry.get("overall_score")
    criteria = entry.get("criterion_scores", {})

    if label not in SCORE_RANGES:
        return False, f"unknown label '{label}'"
    if score is None:
        return False, "missing overall_score"
    score = int(score)
    lo, hi = SCORE_RANGES[label]
    if not (lo <= score <= hi):
        return False, f"score {score} out of range for label {label} ({lo}-{hi})"
    missing_keys = [k for k in CRITERIA_KEYS if k not in criteria]
    if missing_keys:
        return False, f"missing criteria keys: {missing_keys}"
    return True, "ok"


def load_responses(responses_dir: Path, zone_filter: str | None, min_ratio: float) -> dict:
    """Load all response files and return relabel_map: pair_id -> entry."""
    relabel_map: dict[str, dict] = {}
    files = sorted(responses_dir.glob("relabel_*.json"))
    if zone_filter:
        files = [f for f in files if zone_filter in f.name]

    loaded_files = 0
    skipped_files = 0
    invalid_entries = 0

    for rf in files:
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {rf.name}: could not parse — {e}")
            skipped_files += 1
            continue

        expected  = data.get("pair_count", 10)
        relabels  = data.get("relabels", [])
        ratio     = len(relabels) / expected if expected else 0

        if ratio < min_ratio:
            print(f"  SKIP {rf.name}: only {len(relabels)}/{expected} pairs parsed ({ratio:.0%} < {min_ratio:.0%})")
            skipped_files += 1
            continue

        for entry in relabels:
            normalize_entry(entry)
            valid, reason = validate_relabel(entry)
            if not valid:
                invalid_entries += 1
                print(f"  INVALID {entry.get('pair_id','?')} in {rf.name}: {reason}")
                continue
            pid = entry["pair_id"]
            relabel_map[pid] = entry

        loaded_files += 1

    print(f"  Response files: {loaded_files} loaded, {skipped_files} skipped")
    print(f"  Valid relabels: {len(relabel_map)}  |  invalid entries: {invalid_entries}")
    return relabel_map


def main():
    parser = argparse.ArgumentParser(description="Apply LLM relabels to cv_jd_pairs.jsonl")
    parser.add_argument("--pairs",     default=str(PAIRS_DEFAULT), help="Path to cv_jd_pairs.jsonl")
    parser.add_argument("--responses", default=str(RESPONSES_DIR), help="Path to responses directory")
    parser.add_argument("--zone",      default=None, help="Only apply one zone, e.g. weak_moderate_60")
    parser.add_argument("--dry-run",   action="store_true", help="Preview changes without writing")
    parser.add_argument("--min-parsed-ratio", type=float, default=0.7,
                        help="Skip response files where < N fraction of pairs parsed (default: 0.7)")
    parser.add_argument("--show-boundary-warnings", action="store_true",
                        help="Warn when new overall_score still falls in a boundary zone")
    args = parser.parse_args()

    pairs_path     = Path(args.pairs)
    responses_dir  = Path(args.responses)

    # Load relabels
    print(f"Loading responses from {responses_dir} ...")
    relabel_map = load_responses(responses_dir, args.zone, args.min_parsed_ratio)

    if not relabel_map:
        raise SystemExit("No valid relabels found. Nothing to apply.")

    # Load pairs
    pairs = []
    with open(pairs_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"\nLoaded {len(pairs)} pairs from {pairs_path}")

    # Apply relabels
    changed        = 0
    label_changed  = 0
    score_changed  = 0
    boundary_warn  = 0

    print(f"\nApplying relabels (relabel_map has {len(relabel_map)} entries) ...")
    for pair in pairs:
        pid = pair["id"]
        if pid not in relabel_map:
            continue

        new = relabel_map[pid]
        new_label    = new["label"]
        new_score    = int(new["overall_score"])
        new_criteria = {k: int(v) for k, v in new["criterion_scores"].items()}
        new_notes    = new.get("label_notes", "")

        old_label = pair.get("label")
        old_score = pair.get("overall_score")

        # Boundary warning (should be rare with the updated rubric)
        if args.show_boundary_warnings and in_boundary_zone(new_score):
            boundary_warn += 1
            print(f"  BOUNDARY WARN {pid}: new score {new_score} still in boundary zone")

        pair["overall_score"]   = new_score
        pair["label"]           = new_label
        pair["criterion_scores"] = new_criteria
        pair["label_notes"]     = new_notes
        pair["labeled_by"]      = "llm_relabeled"
        pair["label_version"]   = "rubric_v0.2"

        changed += 1
        if old_label != new_label:
            label_changed += 1
        if old_score != new_score:
            score_changed += 1

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Pairs updated       : {changed}")
    print(f"  Label changed       : {label_changed}")
    print(f"  Score changed       : {score_changed}")
    print(f"  Unchanged pairs     : {len(pairs) - changed}")
    if boundary_warn:
        print(f"  Boundary warnings   : {boundary_warn} (scores still near edges)")

    if args.dry_run:
        print("\n(dry-run — no files written)")
        return

    # Backup
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = pairs_path.with_suffix(f".backup_{timestamp}.jsonl")
    shutil.copy2(pairs_path, backup_path)
    print(f"\nBackup: {backup_path.name}")

    # Write updated pairs
    with open(pairs_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Written: {pairs_path} ({len(pairs)} pairs, {changed} updated)")
    print(f"\nNext steps:")
    print(f"  1. Validate: python training/validate_dataset.py --dataset-root datasets")
    print(f"  2. Create versioned snapshot when all zones are applied")


if __name__ == "__main__":
    main()
