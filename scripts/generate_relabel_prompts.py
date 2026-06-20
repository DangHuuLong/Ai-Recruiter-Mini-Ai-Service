"""
Generate LLM relabeling prompts for boundary-zone CV-JD pairs.

Boundary zones (+-5 around rubric thresholds at 40 / 60 / 75 / 90):
  poor/weak       : score 35-45
  weak/moderate   : score 55-65
  moderate/strong : score 70-80
  strong/excellent: score 85-95

Usage:
  # Working dataset (default)
  python scripts/generate_relabel_prompts.py

  # Versioned snapshot
  python scripts/generate_relabel_prompts.py --version v0.2

  # Filter options
  python scripts/generate_relabel_prompts.py --zone weak_moderate_60 --split train
  python scripts/generate_relabel_prompts.py --batch-size 5

Outputs ready-to-paste prompt files to artifacts/relabeling/prompts/.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Working dataset (7000 pairs)
PAIRS_DEFAULT   = Path("datasets/processed/cv_jd_pairs.jsonl")
RESUMES_DEFAULT = Path("datasets/raw/resumes.jsonl")
JDS_DEFAULT     = Path("datasets/raw/job_descriptions.jsonl")

OUTPUT_DIR = Path("artifacts/relabeling/prompts")

BOUNDARY_ZONES = {
    "poor_weak_40":        (35, 45),
    "weak_moderate_60":    (55, 65),
    "moderate_strong_75":  (70, 80),
    "strong_excellent_90": (85, 95),
}

DEFAULT_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Rubric block (pasted at top of every batch file)
# ---------------------------------------------------------------------------

RUBRIC_BLOCK = """\
# RELABELING TASK — CV-JD Matching

You are a senior technical recruiter relabeling CV-JD pairs that have ambiguous
scores near rubric boundaries. These pairs were flagged because the original score
landed within +-5 points of a category threshold (40 / 60 / 75 / 90), meaning the
label could reasonably be either category. Your job is to make a CLEAR, DECISIVE
label — not to guess what the original annotator meant.

IGNORE `current_score`. Score from evidence only.

## Two-step scoring process

STEP 1 — Decide the label first.
Read the JD requirements and the resume evidence. Ask:
- Does the candidate cover the REQUIRED skills with real evidence (projects/work)?
- Does their experience level fit the JD level?
- Is there a CLEAR reason to prefer one label over the adjacent one?

STEP 2 — Score each criterion to MATCH the label.
Once you have decided the label, score each criterion so that the weighted total
falls CLEARLY within that category — away from the boundary edges.

Target score ranges (avoid boundary edges):
| label            | overall_score target |
|------------------|---------------------|
| poor_match       | 15–34               |
| weak_match       | 43–57               |
| moderate_match   | 63–72               |
| strong_match     | 78–87               |
| excellent_match  | 92–98               |

If you genuinely cannot distinguish between two adjacent labels from the evidence,
pick the LOWER label and note why the evidence was insufficient.

## Rubric (rubric_v0.2)

| Criterion               | Weight | What to evaluate |
|-------------------------|-------:|-----------------|
| SKILLS_MATCH            |    35% | Required/preferred skill coverage WITH evidence in projects or work — NOT keyword lists alone. |
| EXPERIENCE_RELEVANCE    |    30% | Experience depth, years, role level, responsibility fit for the JD level. |
| PROJECT_RELEVANCE       |    15% | Projects matching the JD stack, domain, or problem type. |
| EDUCATION_CERTIFICATION |    10% | Degree, field, certs, or equivalent background. Neutral ~70 when JD has no requirement. |
| KEYWORD_DOMAIN_ALIGNMENT|    10% | Domain keywords, responsibilities, business/technical context overlap. |

overall_score = SKILLS_MATCH*0.35 + EXPERIENCE_RELEVANCE*0.30 + PROJECT_RELEVANCE*0.15 + EDUCATION_CERTIFICATION*0.10 + KEYWORD_DOMAIN_ALIGNMENT*0.10
(round to nearest integer)

## Level-aware rules

| JD level | Experience expectation |
|----------|----------------------|
| intern   | Academic/personal projects count as experience evidence |
| junior   | 0-2 years or several owned real projects |
| middle   | 2-4 years, strong project ownership |
| senior   | 4+ years, leadership/architecture/mentoring evidence |
| unknown  | Infer conservatively from JD responsibilities |

Overqualified candidates: do NOT lower technical scores. Note overqualification in label_notes only.

## Label mapping

| Score  | label            |
|--------|-----------------|
| 90-100 | excellent_match  |
| 75-89  | strong_match     |
| 60-74  | moderate_match   |
| 40-59  | weak_match       |
| 0-39   | poor_match       |

## Output format

Output one JSON object per pair in a ```json code block. Nothing else between pairs.

```json
{
  "pair_id": "pair_xxx",
  "criterion_scores": {
    "SKILLS_MATCH": 0,
    "EXPERIENCE_RELEVANCE": 0,
    "PROJECT_RELEVANCE": 0,
    "EDUCATION_CERTIFICATION": 0,
    "KEYWORD_DOMAIN_ALIGNMENT": 0
  },
  "overall_score": 0,
  "label": "poor_match",
  "label_notes": "Strongest match: [reason]. Key gap: [what is missing]."
}
```

Rules:
- Decide LABEL first, then score criteria to match
- overall_score = weighted formula (rounded), must fall in the label's range
- overall_score must NOT be in the edge zones: 35-45, 55-65, 70-80, 85-95
- label_notes: one sentence on strongest match evidence, one sentence on key gap
- Do NOT anchor on current_score or current_notes

---
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> dict:
    records = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                records[obj["id"]] = obj
    return records


def boundary_zone(score: int) -> str | None:
    for zone_name, (lo, hi) in BOUNDARY_ZONES.items():
        if lo <= score <= hi:
            return zone_name
    return None


def fmt_list(items, max_n: int = 12) -> str:
    if not items:
        return "(none)"
    items = [str(x) for x in items]
    if len(items) > max_n:
        return ", ".join(items[:max_n]) + f" (+{len(items) - max_n} more)"
    return ", ".join(items)


def fmt_experience(exp_list) -> str:
    if not exp_list:
        return "  (none)"
    lines = []
    for e in (exp_list if isinstance(exp_list, list) else [])[:5]:
        if isinstance(e, dict):
            role     = e.get("role") or e.get("title") or "?"
            company  = e.get("company", "?")
            duration = e.get("duration") or e.get("dates") or e.get("period") or ""
            lines.append(f"  • {role} @ {company}  {duration}".rstrip())
        else:
            lines.append(f"  • {str(e)[:120]}")
    return "\n".join(lines)


def fmt_projects(proj_list) -> str:
    if not proj_list:
        return "  (none)"
    lines = []
    for p in (proj_list if isinstance(proj_list, list) else [])[:5]:
        if isinstance(p, dict):
            name     = p.get("name", "?")
            desc     = (p.get("description") or "")[:150]
            tech     = p.get("technologies", [])
            tech_str = f"  [{', '.join(tech[:6])}]" if tech else ""
            lines.append(f"  • {name}: {desc}{tech_str}")
        else:
            lines.append(f"  • {str(p)[:150]}")
    return "\n".join(lines)


def fmt_education(edu) -> str:
    if not edu:
        return "  (none)"
    if isinstance(edu, dict):
        degree = edu.get("degree", "?")
        field  = edu.get("field_of_study", "?")
        status = edu.get("status", "")
        return f"  {degree} in {field}" + (f" ({status})" if status else "")
    if isinstance(edu, list):
        lines = []
        for e in edu[:3]:
            if isinstance(e, dict):
                lines.append(f"  • {e.get('degree','?')} in {e.get('field_of_study','?')}")
            else:
                lines.append(f"  • {str(e)[:100]}")
        return "\n".join(lines)
    return f"  {str(edu)[:150]}"


def fmt_pair_block(pair: dict, resume: dict, jd: dict) -> str:
    score    = pair["overall_score"]
    zone     = boundary_zone(score)
    pair_id  = pair["id"]
    crit     = pair.get("criterion_scores", {})
    crit_str = "  " + "  ".join(f"{k}={v}" for k, v in crit.items())

    requirements    = jd.get("requirements", [])
    responsibilities = jd.get("responsibilities", [])
    req_text  = "\n".join(f"    - {r}" for r in requirements[:10])     or "    (none)"
    resp_text = "\n".join(f"    - {r}" for r in responsibilities[:8])  or "    (none)"

    summary = (resume.get("summary") or "")[:250]

    return f"""\
{'='*68}
PAIR {pair_id}
  current_score   : {score}  (zone: {zone})
  current_label   : {pair.get('label', '?')}
  current_notes   : {pair.get('label_notes', '(none)')}
  current_criteria:
{crit_str}

JOB DESCRIPTION: {jd.get('title', '?')}  [{jd.get('level', '?')}]
  Required skills    : {fmt_list(jd.get('required_skills', []))}
  Preferred skills   : {fmt_list(jd.get('preferred_skills', []))}
  Min experience     : {jd.get('min_experience_years', '?')} years
  Education req      : {jd.get('education_requirement', 'none')}
  Requirements:
{req_text}
  Responsibilities:
{resp_text}

RESUME: [{resume.get('candidate_level', '?')}]  {resume.get('experience_years', '?')} yrs experience
  Summary  : {summary if summary else '(none)'}
  Skills   : {fmt_list(resume.get('skills', []))}
  Experience:
{fmt_experience(resume.get('experience', []))}
  Projects:
{fmt_projects(resume.get('projects', []))}
  Education:
{fmt_education(resume.get('education'))}
  Certifications: {fmt_list(resume.get('certifications', []))}
{'='*68}

"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LLM relabeling prompt files")
    parser.add_argument("--version",    default=None, help="Use versioned snapshot, e.g. v0.2 (datasets/versions/<v>)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Pairs per prompt file")
    parser.add_argument("--zone",  default=None, help="Only one zone, e.g. weak_moderate_60")
    parser.add_argument("--split", default=None, help="Filter by split: train / validation / test")
    args = parser.parse_args()

    batch_size = args.batch_size

    if args.version:
        version_dir = Path("datasets/versions") / args.version
        pairs_path   = version_dir / "cv_jd_pairs.jsonl"
        resumes_path = version_dir / "resumes.jsonl"
        jds_path     = version_dir / "job_descriptions.jsonl"
        source_label = str(version_dir)
    else:
        pairs_path   = PAIRS_DEFAULT
        resumes_path = RESUMES_DEFAULT
        jds_path     = JDS_DEFAULT
        source_label = "working dataset (datasets/raw + datasets/processed)"

    print(f"Loading {source_label} ...")
    pairs   = load_jsonl(pairs_path)
    resumes = load_jsonl(resumes_path)
    jds     = load_jsonl(jds_path)
    print(f"  {len(pairs)} pairs  |  {len(resumes)} resumes  |  {len(jds)} JDs")

    # Find boundary pairs
    boundary_pairs: dict[str, list] = defaultdict(list)
    skipped_missing = 0
    for pair in pairs.values():
        score = pair.get("overall_score", -1)
        zone  = boundary_zone(score)
        if not zone:
            continue
        if args.zone and zone != args.zone:
            continue
        if args.split and pair.get("split") != args.split:
            continue
        resume_id = pair.get("resume_id")
        jd_id     = pair.get("job_description_id")
        if resume_id not in resumes or jd_id not in jds:
            skipped_missing += 1
            continue
        boundary_pairs[zone].append(pair)

    # Summary
    total = sum(len(v) for v in boundary_pairs.values())
    print(f"\n{'-'*52}")
    print(f"Boundary pairs  (zone={args.zone or 'all'}, split={args.split or 'all'})")
    print(f"{'-'*52}")
    for zone_name in sorted(BOUNDARY_ZONES):
        zone_list = boundary_pairs.get(zone_name, [])
        if not zone_list:
            continue
        scores = [p["overall_score"] for p in zone_list]
        print(f"  {zone_name:32s}: {len(zone_list):4d} pairs  scores {min(scores):3d}-{max(scores):3d}")
    print(f"  {'TOTAL':32s}: {total:4d} pairs")
    if skipped_missing:
        print(f"  (skipped {skipped_missing} pairs - missing resume/JD records)")

    batch_count = -(-total // batch_size)
    print(f"\nBatch size: {batch_size} pairs/file -> {batch_count} prompt files")

    if total == 0:
        print("No boundary pairs found. Nothing to generate.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_index  = []
    total_files = 0

    for zone_name in sorted(BOUNDARY_ZONES):
        zone_list = boundary_pairs.get(zone_name, [])
        if not zone_list:
            continue
        zone_list_sorted = sorted(zone_list, key=lambda p: p["overall_score"])

        for batch_num, i in enumerate(range(0, len(zone_list_sorted), batch_size), start=1):
            batch    = zone_list_sorted[i : i + batch_size]
            pair_ids = [p["id"] for p in batch]

            pair_blocks = "".join(
                fmt_pair_block(p, resumes[p["resume_id"]], jds[p["job_description_id"]])
                for p in batch
            )

            content = (
                f"# Relabeling — {zone_name} — batch {batch_num}\n"
                f"# Pairs ({len(batch)}): {', '.join(pair_ids)}\n\n"
                + RUBRIC_BLOCK
                + f"## PAIRS TO RELABEL  ({len(batch)} pairs)\n\n"
                + pair_blocks
                + f"\n## OUTPUT — paste {len(batch)} JSON objects below\n\n"
            )

            fname    = f"relabel_{zone_name}_batch{batch_num:02d}.txt"
            out_path = OUTPUT_DIR / fname
            out_path.write_text(content, encoding="utf-8")
            file_index.append((fname, zone_name, batch_num, pair_ids))
            total_files += 1

    # Write index
    index_lines = [
        "# Relabeling Prompt Index\n\n",
        f"- Dataset: `{source_label}`\n",
        f"- Total boundary pairs: {total}\n",
        f"- Batch size: {batch_size}\n\n",
        "| File | Zone | Batch | Pairs |\n",
        "|------|------|------:|-------|\n",
    ]
    for fname, zone, bnum, ids in file_index:
        index_lines.append(f"| `{fname}` | {zone} | {bnum} | {', '.join(ids)} |\n")

    (OUTPUT_DIR / "INDEX.md").write_text("".join(index_lines), encoding="utf-8")

    print(f"\n{'-'*52}")
    print(f"Output : {OUTPUT_DIR.resolve()}")
    print(f"Files  : {total_files} prompt files  +  INDEX.md")
    print(f"\nWorkflow:")
    print(f"  1. Open a file in {OUTPUT_DIR}/")
    print(f"  2. Copy entire content -> paste into Claude or GPT-4")
    print(f"  3. Save LLM JSON responses to artifacts/relabeling/responses/<same_filename>")
    print(f"  4. Run scripts/apply_relabels.py to write updated labels back to dataset")


if __name__ == "__main__":
    main()
