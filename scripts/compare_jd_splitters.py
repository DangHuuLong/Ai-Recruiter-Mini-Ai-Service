"""
Diagnostic tool: compare the regex-only JD section splitter against the
ML-assisted splitter on the same input text(s). Mirrors
scripts/compare_splitters.py (the CV version).

Read-only — does not modify any files, not part of the served pipeline.
Requires the JD section-classifier model to actually be enabled (via
JD_SECTION_CLASSIFIER_FALLBACK_MODE=model_with_regex_fallback and a valid
JD_SECTION_CLASSIFIER_MODEL_PATH), otherwise there's nothing to compare.

Usage:
  python scripts/compare_jd_splitters.py --input datasets/jd_section_splitting/v0.1/labeled_lines.jsonl
  python scripts/compare_jd_splitters.py --text-file some_jd.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.parsers.job_description_parser import _ml_assisted_split, _regex_split_sections  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _line_owner(sections: dict[str, list[str]]) -> dict[str, str]:
    owner: dict[str, str] = {}
    for label, lines in sections.items():
        for line in lines:
            owner[line] = label
    return owner


def compare_one(raw_text: str, doc_id: str) -> dict:
    regex_sections = _regex_split_sections(raw_text)
    try:
        ml_sections = _ml_assisted_split(raw_text)
    except Exception as e:
        print(f"  [!] ML split unavailable ({e}) — nothing to compare. "
              f"Check JD_SECTION_CLASSIFIER_FALLBACK_MODE / JD_SECTION_CLASSIFIER_MODEL_PATH.")
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
        print(f"\n  {doc_id}: {len(changed_lines)} line(s) changed section")
        for line, old, new in changed_lines:
            preview = line if len(line) <= 70 else line[:67] + "..."
            print(f"    [{old!r:>16} -> {new!r:<16}]  {preview}")
    else:
        print(f"\n  {doc_id}: no change")

    return {
        "doc_id": doc_id,
        "n_changed_lines": len(changed_lines),
        "other_before": len(regex_sections.get("other", [])),
        "other_after": len(ml_sections.get("other", [])),
    }


def compare_batch(input_path: Path) -> None:
    docs = _load_jsonl(input_path)
    results = []
    for doc in docs:
        lines = doc.get("lines", [])
        raw_text = "\n".join(item.get("text", "") for item in lines if isinstance(item, dict))
        if not raw_text:
            continue
        results.append(compare_one(raw_text, doc.get("id", "?")))

    total_docs = len(results)
    changed_docs = sum(1 for r in results if r["n_changed_lines"] > 0)
    total_other_before = sum(r["other_before"] for r in results)
    total_other_after = sum(r["other_after"] for r in results)

    print("\n" + "=" * 60)
    print(f"  JDs compared        : {total_docs}")
    print(f"  JDs changed         : {changed_docs}")
    print(f"  'other' lines before: {total_other_before}")
    print(f"  'other' lines after : {total_other_after}  "
          f"({total_other_before - total_other_after} moved out of 'other')")
    print("=" * 60)


def compare_text_file(text_path: Path) -> None:
    raw_text = text_path.read_text(encoding="utf-8-sig")
    compare_one(raw_text, doc_id=text_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare regex-only vs ML-assisted JD section splitting.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="labeled_lines.jsonl-style dataset to batch-compare.")
    group.add_argument("--text-file", type=Path, help="Single raw JD text file to compare.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        compare_batch(args.input)
    else:
        compare_text_file(args.text_file)


if __name__ == "__main__":
    main()
