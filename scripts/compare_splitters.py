"""
Diagnostic tool: compare the regex-only section splitter against the
ML-assisted splitter on the same input text(s), and print a line-level diff.

Read-only — does not modify any files, not part of the served pipeline.
Requires the section-classifier model to actually be enabled (via
SECTION_CLASSIFIER_FALLBACK_MODE=model_with_regex_fallback and a valid
SECTION_CLASSIFIER_MODEL_PATH pointing at trained artifacts), otherwise
there's nothing to compare — the ML path will just raise and there'd be no
diff to show.

Usage:
  # Compare over every resume in a labeled_lines.jsonl-style file (raw text
  # reconstructed from each resume's `lines`, ignoring the true labels —
  # this is purely a diagnostic on the SPLITTER, not a labeled-data eval;
  # for a real accuracy number, use training/evaluate_section_classifier.py):
  python scripts/compare_splitters.py --input datasets/section_splitting/v0.1/labeled_lines.jsonl

  # Compare a single ad hoc resume text file:
  python scripts/compare_splitters.py --text-file some_resume.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.parsers.section_splitter import _ml_assisted_split, _regex_split_sections  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _line_owner(sections: dict[str, str]) -> dict[str, str]:
    """Map each distinct line of text -> the section it ended up in (last
    write wins if the exact same line text appears in multiple sections)."""
    owner: dict[str, str] = {}
    for label, text in sections.items():
        for line in text.splitlines():
            owner[line] = label
    return owner


def compare_one(raw_text: str, resume_id: str) -> dict:
    regex_sections = _regex_split_sections(raw_text)
    try:
        ml_sections = _ml_assisted_split(raw_text)
    except Exception as e:
        print(f"  [!] ML split unavailable ({e}) — nothing to compare. "
              f"Check SECTION_CLASSIFIER_FALLBACK_MODE / SECTION_CLASSIFIER_MODEL_PATH.")
        raise SystemExit(1)

    regex_owner = _line_owner(regex_sections)
    ml_owner = _line_owner(ml_sections)

    changed_lines = []
    for line in regex_owner:
        old = regex_owner.get(line)
        new = ml_owner.get(line)
        if old != new:
            changed_lines.append((line, old, new))

    if changed_lines:
        print(f"\n  {resume_id}: {len(changed_lines)} line(s) changed section")
        for line, old, new in changed_lines:
            preview = line if len(line) <= 70 else line[:67] + "..."
            print(f"    [{old!r:>16} -> {new!r:<16}]  {preview}")
    else:
        print(f"\n  {resume_id}: no change")

    other_before = len(regex_sections.get("other", "").splitlines())
    other_after = len(ml_sections.get("other", "").splitlines())
    return {
        "resume_id": resume_id,
        "n_changed_lines": len(changed_lines),
        "other_before": other_before,
        "other_after": other_after,
    }


def compare_batch(input_path: Path) -> None:
    resumes = _load_jsonl(input_path)
    results = []
    for resume in resumes:
        lines = resume.get("lines", [])
        raw_text = "\n".join(item.get("text", "") for item in lines if isinstance(item, dict))
        if not raw_text:
            continue
        results.append(compare_one(raw_text, resume.get("id", "?")))

    total_resumes = len(results)
    changed_resumes = sum(1 for r in results if r["n_changed_lines"] > 0)
    total_other_before = sum(r["other_before"] for r in results)
    total_other_after = sum(r["other_after"] for r in results)

    print("\n" + "=" * 60)
    print(f"  Resumes compared   : {total_resumes}")
    print(f"  Resumes changed    : {changed_resumes}")
    print(f"  'other' lines before: {total_other_before}")
    print(f"  'other' lines after : {total_other_after}  "
          f"({total_other_before - total_other_after} moved out of 'other')")
    print("=" * 60)


def compare_text_file(text_path: Path) -> None:
    raw_text = text_path.read_text(encoding="utf-8-sig")
    compare_one(raw_text, resume_id=text_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare regex-only vs ML-assisted section splitting.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="labeled_lines.jsonl-style dataset to batch-compare.")
    group.add_argument("--text-file", type=Path, help="Single raw resume text file to compare.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        compare_batch(args.input)
    else:
        compare_text_file(args.text_file)


if __name__ == "__main__":
    main()
