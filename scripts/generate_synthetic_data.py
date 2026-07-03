"""
Synthetic data generator for CV-JD pairs.

Workflow per batch (9 pairs from 3 resumes × 3 JDs):
  [1] Generate CV+JD prompt  → copy to LLM → paste output to scripts/temp_input.txt
  [2] Save result            → if waiting for CV+JD: saves resumes+JDs, auto-shows pair prompt
                             → if waiting for pairs:  validates + saves pairs, shows updated stats
  [0] Exit

Run:
  python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPTS_DIR  = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent

RESUMES_PATH = PROJECT_ROOT / "datasets" / "raw_v2" / "resumes.jsonl"
JDS_PATH     = PROJECT_ROOT / "datasets" / "raw_v2" / "job_descriptions.jsonl"
PAIRS_PATH   = PROJECT_ROOT / "datasets" / "raw_v2" / "cv_jd_pairs.jsonl"
TEMP_INPUT     = SCRIPTS_DIR / "temp_input.txt"
PROMPT_OUTPUT  = SCRIPTS_DIR / "prompt_output.txt"
SESSION_FILE   = SCRIPTS_DIR / ".synthetic_session.json"

# ── batch size constants ───────────────────────────────────────────────────────

N_RESUMES = 3
N_JDS     = 3
N_PAIRS   = N_RESUMES * N_JDS  # 9

# ── constants ─────────────────────────────────────────────────────────────────

BOUNDARY_ZONES   = [(35, 45), (55, 65), (70, 80), (85, 95)]
FORBIDDEN_LABELS: set[str] = set()
ALLOWED_LABELS   = {"poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"}

SCORE_RANGES = {
    "poor_match":      (0,  39),
    "weak_match":      (40, 59),
    "moderate_match":  (60, 74),
    "strong_match":    (75, 89),
    "excellent_match": (90, 100),
}

# Valid score windows after removing boundary zones
VALID_SCORE_WINDOWS = {
    "poor_match":      (20, 34),   # 35-45 is forbidden boundary zone
    "weak_match":      (46, 54),
    "moderate_match":  (66, 69),
    "strong_match":    (81, 84),
    "excellent_match": (96, 100),
}

CRITERIA_KEYS = [
    "SKILLS_MATCH",
    "EXPERIENCE_RELEVANCE",
    "PROJECT_RELEVANCE",
    "EDUCATION_CERTIFICATION",
    "KEYWORD_DOMAIN_ALIGNMENT",
]

ALLOWED_LEVELS    = {"intern", "junior", "middle", "senior", "unknown"}
ALLOWED_LANGUAGES = {"en", "vi", "mixed"}

CV_MIN_WORDS = 250
JD_MIN_WORDS = 180

# ── domain list ────────────────────────────────────────────────────────────────

DOMAINS = [
    # IT
    "Backend API Development (Python, FastAPI, Node.js, PostgreSQL)",
    "Frontend Development (React, TypeScript, Next.js, Tailwind CSS)",
    "Data Science and Machine Learning (Python, scikit-learn, PyTorch, pandas)",
    "DevOps and Cloud Infrastructure (AWS, Docker, Kubernetes, Terraform)",
    "Mobile Development (React Native, Flutter, Swift, Kotlin)",
    "Full-Stack Web Development (Python backend + React frontend)",
    "AI and NLP Engineering (Transformers, LangChain, RAG, embeddings)",
    "Cybersecurity and Information Security (pentesting, SIEM, network)",
    "Database Engineering (PostgreSQL, MongoDB, Redis, data modeling)",
    "Product Management in Technology (roadmap, agile, stakeholders)",
    # Non-IT
    "Digital Marketing and Content Strategy (SEO, social media, analytics)",
    "Finance and Accounting in Fintech (financial modeling, reporting, compliance)",
    "Human Resources and Talent Acquisition (recruiting, HRIS, onboarding)",
    "Supply Chain and Logistics Management (procurement, inventory, ERP)",
    "UI/UX Design and Product Design (Figma, user research, prototyping)",
    "Business Analysis and Project Management (requirements, BPMN, stakeholders)",
    "E-commerce and Retail Management (marketplace, operations, merchandising)",
    "Healthcare Administration and Medical Coordination (clinical ops, EMR)",
]

# ── JSONL helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_max_numeric_id(records: list[dict], prefix: str) -> int:
    max_id = 0
    for r in records:
        rid = r.get("id", "")
        if rid.startswith(prefix):
            try:
                max_id = max(max_id, int(rid[len(prefix):]))
            except ValueError:
                pass
    return max_id

# ── session ────────────────────────────────────────────────────────────────────

def load_session() -> dict:
    if SESSION_FILE.exists():
        try:
            return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"state": "idle"}


def save_session(data: dict) -> None:
    SESSION_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# ── stats ──────────────────────────────────────────────────────────────────────

def show_stats() -> None:
    resumes = load_jsonl(RESUMES_PATH)
    jds     = load_jsonl(JDS_PATH)
    pairs   = load_jsonl(PAIRS_PATH)
    dist    = Counter(p.get("label") for p in pairs)
    total   = len(pairs)

    labels_order = ["poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"]
    print("\n" + "=" * 60)
    print(f"  Resumes: {len(resumes)}   JDs: {len(jds)}   Pairs: {total}")
    print(f"  Batch size: {N_RESUMES} resumes × {N_JDS} JDs = {N_PAIRS} pairs")
    print("  Label distribution:")
    for label in labels_order:
        count = dist.get(label, 0)
        pct   = count / total * 100 if total else 0
        bar   = "█" * int(pct / 2.5)
        print(f"    {label:<20} {count:>5}  ({pct:4.1f}%)  {bar}")

    labels_order_all = ["poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"]
    target = max((dist.get(label, 0) for label in ALLOWED_LABELS), default=0)
    total_need = 0
    print(f"\n  Still needed to balance (target = {target} each):")
    for label in labels_order_all:
        need = max(0, target - dist.get(label, 0))
        total_need += need
        batches = need // N_PAIRS + (1 if need % N_PAIRS else 0)
        print(f"    {label:<20} +{need:>4}  (~{batches:>3} batches)")
    batches_need = total_need // N_PAIRS + (1 if total_need % N_PAIRS else 0)
    print(f"    Total new pairs needed: ~{total_need}  (~{batches_need} batches)")
    print("=" * 60)


def compute_label_targets(n_pairs: int = N_PAIRS) -> dict[str, int]:
    """Return how many of each allowed label to target in a batch.

    Labels with fewer existing pairs receive more weight (inverse-count weighting).
    """
    pairs = load_jsonl(PAIRS_PATH)
    dist  = Counter(p.get("label") for p in pairs if p.get("label") in ALLOWED_LABELS)

    labels = sorted(ALLOWED_LABELS)  # stable order

    if not dist:
        per       = n_pairs // len(labels)
        remainder = n_pairs - per * len(labels)
        targets   = {label: per for label in labels}
        for i, label in enumerate(labels):
            if i < remainder:
                targets[label] += 1
        return targets

    # If the single most-deficient label is behind the NEXT-most-deficient one by
    # a full batch's worth or more, splitting this batch 1-1-2-2-3 across all 5
    # labels (as inverse-weighting below would do) can't meaningfully close that
    # gap and just keeps stalling the worst-off label. Dedicate the whole batch
    # to it instead — e.g. poor_match=0 while every other label is already 27+
    # means all 9 pairs this batch should be poor_match, not 1-of-9.
    by_count = sorted(labels, key=lambda l: dist.get(l, 0))
    neediest, second = by_count[0], by_count[1]
    if dist.get(second, 0) - dist.get(neediest, 0) >= n_pairs:
        return {label: (n_pairs if label == neediest else 0) for label in labels}

    # Inverse weight: label with fewer pairs gets higher weight
    max_count = max(dist.get(label, 0) for label in labels)
    weights   = {label: max_count - dist.get(label, 0) + 1 for label in labels}
    total_w   = sum(weights.values())

    targets: dict[str, int] = {}
    remaining = n_pairs
    for i, label in enumerate(sorted(labels, key=lambda l: weights[l], reverse=True)):
        if i == len(labels) - 1:
            targets[label] = max(1, remaining)
        else:
            count = max(1, round(n_pairs * weights[label] / total_w))
            targets[label] = count
            remaining -= count
    return targets


LEVELS = ["junior", "middle", "senior"]

LEVEL_INFO = {
    "junior": ("0-1 yr, fresh graduate or first job",            "min 1 yr, entry/associate level role"),
    "middle": ("3-5 yrs total experience",                        "min 3 yrs, mid-level individual contributor"),
    "senior": ("6-9 yrs, tech lead or principal",                 "min 6 yrs, lead or senior engineer, deep specialization required"),
}


SKILL_OVERLAP_TARGET = {
    "poor_match":      (10, 30),
    "weak_match":      (35, 55),
    "moderate_match":  (55, 70),
    "strong_match":    (70, 85),
    "excellent_match": (85, 100),
}


def natural_label_for_gap(gap: int) -> str:
    """Label a resume/JD seniority-tier gap (JD tier − resume tier) naturally produces."""
    if gap >= 2:
        return "poor_match"
    if gap == 1:
        return "weak_match"
    if gap == 0:
        return "excellent_match"
    if gap == -1:
        return "strong_match"
    return "moderate_match"  # gap <= -2, heavily overqualified


def _level_partitions(n: int) -> list[tuple[int, int, int]]:
    """All (n_junior, n_middle, n_senior) counts that sum to n."""
    return [
        (j, m, n - j - m)
        for j in range(n + 1)
        for m in range(n + 1 - j)
    ]


def choose_seniority_mix(
    label_targets: dict[str, int],
    n_resumes: int = N_RESUMES,
    n_jds: int = N_JDS,
) -> tuple[list[str], list[str]]:
    """Pick how many junior/middle/senior resumes and JDs to generate this batch.

    A fixed 1-junior/1-middle/1-senior split on both sides always yields the same
    natural label mix (1 poor_match cell, 2 weak_match cells, 2 strong, 3 excellent,
    1 moderate) no matter how skewed the dataset already is — deficits then get
    "fixed" by telling the labeler to bend its scoring, which it can't do honestly.
    Instead, search over seniority-tier counts for BOTH sides and pick whichever mix
    makes the resulting grid's natural label counts closest to label_targets.

    Plain unweighted L1 distance treats "1 pair short on excellent_match (already
    well-stocked)" the same as "1 pair short on poor_match (the neediest label)" —
    ties between a balanced-but-wrong mix and a mix that actually fills the neediest
    label get broken arbitrarily, so poor_match can silently lose the tie. Instead,
    weight each label's error by how deficient it currently is (same inverse-count
    weighting as compute_label_targets) — a "water-filling" priority where closing
    the gap on the lowest label always outweighs precision on already-full ones.
    """
    dist_now  = Counter(p.get("label") for p in load_jsonl(PAIRS_PATH) if p.get("label") in ALLOWED_LABELS)
    max_count = max((dist_now.get(l, 0) for l in ALLOWED_LABELS), default=0)
    weight    = {l: max_count - dist_now.get(l, 0) + 1 for l in ALLOWED_LABELS}

    best_mix = None
    best_score = None
    for r_counts in _level_partitions(n_resumes):
        for j_counts in _level_partitions(n_jds):
            dist: Counter = Counter()
            for r_idx, r_count in enumerate(r_counts):
                if not r_count:
                    continue
                for j_idx, j_count in enumerate(j_counts):
                    if not j_count:
                        continue
                    label = natural_label_for_gap(j_idx - r_idx)
                    dist[label] += r_count * j_count
            score = sum(
                weight[l] * abs(dist.get(l, 0) - label_targets.get(l, 0))
                for l in ALLOWED_LABELS
            )
            if best_score is None or score < best_score:
                best_score = score
                best_mix = (r_counts, j_counts)

    r_counts, j_counts = best_mix
    resume_levels = [lvl for lvl, count in zip(LEVELS, r_counts) for _ in range(count)]
    jd_levels     = [lvl for lvl, count in zip(LEVELS, j_counts) for _ in range(count)]
    return resume_levels, jd_levels

# ── next IDs ───────────────────────────────────────────────────────────────────

def get_next_ids() -> tuple[int, int, int]:
    resume_start = get_max_numeric_id(load_jsonl(RESUMES_PATH), "resume_") + 1
    jd_start     = get_max_numeric_id(load_jsonl(JDS_PATH),     "jd_")     + 1
    pair_start   = get_max_numeric_id(load_jsonl(PAIRS_PATH),   "pair_")   + 1
    return resume_start, jd_start, pair_start

# ── boundary / validation helpers ─────────────────────────────────────────────

def in_boundary_zone(score: int) -> bool:
    return any(lo <= score <= hi for lo, hi in BOUNDARY_ZONES)


def expected_label(score: int) -> str:
    for label, (lo, hi) in SCORE_RANGES.items():
        if lo <= score <= hi:
            return label
    return "unknown"

# ── prompt builders ────────────────────────────────────────────────────────────

def build_cvjd_prompt(
    domain: str,
    resume_start: int,
    jd_start: int,
    resume_levels: list[str] | None = None,
    jd_levels: list[str] | None = None,
) -> str:
    r_ids  = [f"resume_{resume_start + i}" for i in range(N_RESUMES)]
    j_ids  = [f"jd_{jd_start + i}"        for i in range(N_JDS)]

    resume_levels = resume_levels or ["junior", "middle", "senior"][:N_RESUMES]
    jd_levels     = jd_levels or ["junior", "middle", "senior"][:N_JDS]

    resume_lines = "\n".join(
        f"Resume {rid}: {lvl:<8} ({LEVEL_INFO[lvl][0]})"
        for rid, lvl in zip(r_ids, resume_levels)
    )
    jd_lines = "\n".join(
        f"JD {jid}: {lvl:<8} ({LEVEL_INFO[lvl][1]})"
        for jid, lvl in zip(j_ids, jd_levels)
    )

    expected_dist: Counter = Counter()
    for r_lvl in resume_levels:
        for j_lvl in jd_levels:
            gap = LEVELS.index(j_lvl) - LEVELS.index(r_lvl)
            expected_dist[natural_label_for_gap(gap)] += 1
    expected_lines = "\n".join(
        f"  {label:<16} × {count}" for label, count in expected_dist.items()
    )

    overlap_rows = []
    for rid, r_lvl in zip(r_ids, resume_levels):
        for jid, j_lvl in zip(j_ids, jd_levels):
            gap   = LEVELS.index(j_lvl) - LEVELS.index(r_lvl)
            label = natural_label_for_gap(gap)
            lo, hi = SKILL_OVERLAP_TARGET[label]
            overlap_rows.append(f"  {rid} × {jid}  ({label:<16}) → resume should cover {lo}-{hi}% of that JD's required_skills")
    overlap_lines = "\n".join(overlap_rows)

    return f"""You are a dataset generator for an AI recruitment system.
Generate exactly {N_RESUMES} resumes and {N_JDS} job descriptions in the domain: {domain}

These {N_RESUMES + N_JDS} documents will be cross-paired into {N_PAIRS} CV-JD pairs.
Target score range: 20-100 (all 5 labels including poor_match).

════════════════════════════════════
SENIORITY ASSIGNMENT (follow exactly)
════════════════════════════════════
{resume_lines}

{jd_lines}

This mix was chosen (based on which labels the dataset currently lacks) so that
cross-pairing every resume × every JD naturally produces this label spread —
score the resulting pairs honestly instead of forcing a different distribution:
{expected_lines}

Gap rule (JD seniority tier − resume seniority tier), for reference:
  +2 → poor_match (score 20-34)       +1 → weak_match (score 46-54)
   0 → excellent_match (score 96-100) -1 → strong_match (score 81-84)
  <=-2 → moderate_match (score 66-69, heavily overqualified)

════════════════════════════════════
REQUIRED SKILLS OVERLAP — AUTHOR THIS DELIBERATELY, PER PAIR
════════════════════════════════════
A seniority gap alone will NOT produce poor_match or weak_match — if the junior
resume's skills list still covers most of the senior JD's required_skills (common
when both are "same domain"), the honest label stays weak_match/moderate_match no
matter how big the experience gap is. You must also design each resume's skills
list so its overlap with each JD's required_skills roughly matches the target below
(pick which required_skills to give/omit per resume with this table in mind — e.g.
for a poor_match target, have the junior resume know only the basic/common tools
and be missing the JD's specialized or advanced-tier skills entirely):
{overlap_lines}

════════════════════════════════════
CV raw_text — STRICT REQUIREMENTS
════════════════════════════════════
Minimum 280 words per resume. Write as connected prose paragraphs (not bullet points).
Include ALL of the following to reach 280+ words:
  • Job history: for each role — job title, company type (startup/enterprise/agency),
    duration, team size, 2-3 specific responsibilities with concrete tools used
  • Technologies: list specific versions or ecosystems where natural (e.g. "FastAPI 0.100,
    PostgreSQL 14, deployed on AWS ECS")
  • Quantified achievements where possible: latency reduced by X%, served Y requests/day,
    reduced deployment time from X to Y minutes
  • Projects: 1-2 side or internal projects with name, goal, tech stack, outcome
  • Current focus and next career direction (1-2 sentences)
  • Education: degree, field, graduation year

Do NOT include: candidate name, email, phone number, LinkedIn URL, GitHub URL.
Do NOT use bullet points — write connected prose only.

════════════════════════════════════
JD raw_text — STRICT REQUIREMENTS
════════════════════════════════════
Minimum 200 words per job description. Write as connected prose paragraphs.
Include ALL of the following to reach 200+ words:
  • Role overview: what the team does, team size, product context
  • Day-to-day responsibilities (3-4 specific duties with tools/systems named)
  • Technical stack: list every required and preferred technology with context
  • What success looks like in the first 6 months
  • Collaboration: who this person works with (e.g. product, QA, data team)
  • Growth opportunities or scope of impact

Do NOT include: company name. Do NOT use bullet points — write connected prose only.

════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════
Output exactly {N_RESUMES + N_JDS} lines.
Each line is a single JSON object (no array brackets, no commas between lines).
Lines 1-{N_RESUMES}: resumes (IDs: {", ".join(r_ids)})
Lines {N_RESUMES + 1}-{N_RESUMES + N_JDS}: job descriptions (IDs: {", ".join(j_ids)})
No markdown, no explanation — only {N_RESUMES + N_JDS} JSON lines.

RESUME SCHEMA (lines 1-{N_RESUMES}):
{{"id":"<resume_id>","source":"synthetic","candidate_level":"<junior|middle|senior>","raw_text":"<280+ word prose — no names/emails/phones/URLs>","summary":"<one sentence capturing level, domain, and key strength>","skills":["<TitleCased>",...],"normalized_skills":["<snake_case>",...],"experience_years":<float>,"education":{{"degree":"<Bachelor|Master|PhD>","field_of_study":"<field>","status":"<student|final_year|graduate>"}},"projects":[{{"name":"<project name>","description":"<2-3 sentences: goal, tech, outcome>","technologies":["<Tech>",...]}}],"certifications":[],"languages":[{{"name":"English","proficiency":"intermediate"}}],"anonymized":true,"language":"en"}}

JD SCHEMA (lines {N_RESUMES + 1}-{N_RESUMES + N_JDS}):
{{"id":"<jd_id>","source":"synthetic","title":"<Job Title>","level":"<junior|middle|senior>","employment_type":"<internship|full_time|contract>","location":"<City, Country or Remote>","remote_allowed":<true|false>,"posted_date":null,"domain":"<slug>","raw_text":"<200+ word prose — no company name>","responsibilities":["<specific item>",...],"requirements":["<specific item>",...],"nice_to_have":["<specific item>",...],"required_skills":["<TitleCased>",...],"preferred_skills":["<TitleCased>",...],"min_experience_years":<int>,"education_requirement":"<string>","domain_keywords":["<snake_case>",...],"language":"en"}}"""


def build_pair_prompt(
    resumes: list[dict],
    jds: list[dict],
    pair_start: int,
    label_targets: dict[str, int] | None = None,
    current_dist: dict[str, int] | None = None,
) -> str:
    def resume_summary(r: dict) -> str:
        raw_words = len(r.get("raw_text", "").split())
        return (
            f"  id: {r['id']} | level: {r.get('candidate_level')} | "
            f"exp: {r.get('experience_years')} yrs | raw_text: {raw_words} words\n"
            f"  skills: {', '.join(r.get('skills', []))}\n"
            f"  projects: {', '.join(p.get('name','') for p in r.get('projects',[]))}\n"
            f"  background: {r.get('raw_text', '')[:300]}"
        )

    def jd_summary(j: dict) -> str:
        raw_words = len(j.get("raw_text", "").split())
        return (
            f"  id: {j['id']} | title: {j.get('title')} | level: {j.get('level')} | "
            f"min_exp: {j.get('min_experience_years')} yrs | raw_text: {raw_words} words\n"
            f"  required_skills: {', '.join(j.get('required_skills', []))}\n"
            f"  preferred_skills: {', '.join(j.get('preferred_skills', []))}\n"
            f"  description: {j.get('raw_text', '')[:250]}"
        )

    resume_block = "\n\n".join(resume_summary(r) for r in resumes)
    jd_block     = "\n\n".join(jd_summary(j)     for j in jds)

    r_ids = [r["id"] for r in resumes]
    j_ids = [j["id"] for j in jds]

    pair_ids = []
    idx = pair_start
    for rid in r_ids:
        for jid in j_ids:
            pair_ids.append((f"pair_{idx}", rid, jid))
            idx += 1

    pair_order = "\n".join(
        f"  pair_{pair_start + i:<5}  {rid} × {jid}"
        for i, (_, rid, jid) in enumerate(pair_ids)
    )

    label_order = ["poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"]

    # Ground-truth expectation: this batch's resumes/JDs were already generated with a
    # seniority mix chosen (via choose_seniority_mix) to naturally produce the counts the
    # dataset needs. Compute what each pair SHOULD land on from the real seniority gap —
    # this is a target to verify against, not a quota to force onto mismatched pairs.
    expected_dist: Counter = Counter()
    for r in resumes:
        for j in jds:
            r_lvl, j_lvl = r.get("candidate_level"), j.get("level")
            if r_lvl in LEVELS and j_lvl in LEVELS:
                expected_dist[natural_label_for_gap(LEVELS.index(j_lvl) - LEVELS.index(r_lvl))] += 1

    if label_targets or expected_dist:
        total_existing = sum((current_dist or {}).values())

        rows = []
        for label in label_order:
            existing = (current_dist or {}).get(label, 0)
            expected = expected_dist.get(label, 0)
            rows.append(f"  {label:<20}  existing: {existing:>4}  →  expected this batch: {expected} pairs (from seniority gaps below)")

        dist_lines = "\n".join(rows)
        distribution_section = f"""
════════════════════════════════════
LABEL BALANCE — CONTEXT (already engineered into this batch)
════════════════════════════════════
Overall dataset so far: {total_existing} pairs total.
This batch's resume/JD seniority levels were deliberately chosen so that scoring
each pair honestly (per the rubric below) should already land close to this spread —
no need to force it, just verify your scores roughly match:

{dist_lines}

Rules:
• Score every pair from its actual documents — seniority gap, skill overlap, project fit.
• If a pair's honest score lands far from its "expected" label above, trust the
  documents over the expectation and note why in label_notes.
• Adjust criterion scores within valid windows only.
"""
    else:
        distribution_section = ""

    return f"""You are a dataset labeler for an AI recruitment system.
Score {N_PAIRS} CV-JD pairs using rubric v0.2.

════════════════════════════════════
RESUMES
════════════════════════════════════
{resume_block}

════════════════════════════════════
JOB DESCRIPTIONS
════════════════════════════════════
{jd_block}

════════════════════════════════════
SCORING RUBRIC v0.2
════════════════════════════════════
Score each criterion 0-100, then compute weighted overall:
  overall_score = round(
      SKILLS_MATCH            × 0.35 +
      EXPERIENCE_RELEVANCE    × 0.30 +
      PROJECT_RELEVANCE       × 0.15 +
      EDUCATION_CERTIFICATION × 0.10 +
      KEYWORD_DOMAIN_ALIGNMENT× 0.10
  )

Label mapping:
  90-100 → excellent_match
  75-89  → strong_match
  60-74  → moderate_match
  40-59  → weak_match
  0-39   → poor_match

════════════════════════════════════
VALID SCORE WINDOWS (boundary zones removed)
════════════════════════════════════
Use ONLY scores within these windows — no exceptions:
  poor_match:      20-34   (use 25, 28, 30)
  weak_match:      46-54   (use 48, 50, 52)
  moderate_match:  66-69   (use 67, 68)
  strong_match:    81-84   (use 82, 83)
  excellent_match: 96-100  (use 97, 98)

NEVER use scores in boundary zones: 35-45, 55-65, 70-80, 85-95.
These zones are forbidden because they produce ambiguous labels.

════════════════════════════════════
SKILLS_MATCH — DERIVE FROM ACTUAL OVERLAP, DON'T REUSE A NUMBER
════════════════════════════════════
For every pair, first count: matched = required_skills present in the resume's
skills list, total = len(required_skills). Then anchor SKILLS_MATCH to that ratio:
  SKILLS_MATCH ≈ round(100 × matched / total), adjusted ±10 for preferred-skill
  overlap and depth/version alignment.
Two pairs with the same seniority gap can still need different SKILLS_MATCH if
their real overlap ratio differs (e.g. 25% overlap vs 65% overlap must NOT get the
same score just because both are "junior CV × senior JD"). Recompute per pair.

════════════════════════════════════
poor_match SCORING GUIDANCE
════════════════════════════════════
A valid poor_match pair (score 20-34) requires ALL of:
  1. Massive experience gap: intern/fresh grad (0-1 yr) vs senior JD (6+ yrs required)
     → EXPERIENCE_RELEVANCE: 10-25
  2. Core skill mismatch: matched/required skill ratio < 40%
     → SKILLS_MATCH: 15-35 (must equal the computed ratio, not a template number)
  3. Project misalignment: no relevant projects that map to JD responsibilities
     → PROJECT_RELEVANCE: 10-30
Natural source: intern/junior (0-1 yr) CV × senior JD pairing.
Do NOT fabricate poor_match from unrelated domains — the CV and JD must still be in the same broad domain.

DECISION PRIORITY — resolve intern×senior pairs with this BEFORE picking a label:
if resume experience_years <= 1 AND JD min_experience_years >= 6, compute the
skill overlap ratio (matched required_skills / total required_skills) FIRST:
  • ratio < 40%  → this pair MUST be poor_match. The weak_match "experience gap"
    rule below does NOT apply to interns (<=1 yr) — it only applies once the
    candidate has 1-2+ yrs and is past entry level.
  • ratio >= 40% → this pair is weak_match instead (real partial overlap saves it).
This ratio check overrides any temptation to soften the label because the CV and
JD share some domain vocabulary — shared domain alone is not overlap.

════════════════════════════════════
weak_match SCORING GUIDANCE
════════════════════════════════════
A valid weak_match pair (score 46-54) requires at least one of:
  1. Experience gap: resume has >1 yr (already past intern/fresh-grad stage — e.g.
     1-2 yrs) vs senior JD (5+ yrs required). Do NOT use this rule for resumes with
     experience_years <= 1 — see the poor_match DECISION PRIORITY above, which
     routes those to poor_match unless overlap ratio is already >= 40%.
     → EXPERIENCE_RELEVANCE: 25-40 (large gap penalised heavily)
     → SKILLS_MATCH: 45-60 (must equal the actual computed overlap ratio, not a default)
  2. Skill domain mismatch: same broad field but different specialization
     → SKILLS_MATCH: 35-55 (missing core required stack)
     → PROJECT_RELEVANCE: 30-50 (projects don't align with JD responsibilities)

For ALL pairs, criterion scores must be INTERNALLY CONSISTENT:
  • If EXPERIENCE_RELEVANCE is 20, explain in label_notes (e.g. "requires 6+ yrs, candidate has 0.5 yrs")
  • SKILLS_MATCH cannot be 85 if overall is 28 — the math must work
  • Verify: round(SM×0.35 + ER×0.30 + PR×0.15 + EC×0.10 + KD×0.10) == overall_score
{distribution_section}
════════════════════════════════════
PAIRING ORDER ({N_PAIRS} pairs)
════════════════════════════════════
{pair_order}

════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════
Output exactly {N_PAIRS} lines.
Each line is a single JSON object (no array brackets, no commas between lines).
No markdown, no explanation — only {N_PAIRS} JSON lines.

JSON schema per pair:
{{"id":"<pair_id>","resume_id":"<resume_id>","job_description_id":"<jd_id>","split":null,"label":"<poor_match|weak_match|moderate_match|strong_match|excellent_match>","overall_score":<int in valid window>,"criterion_scores":{{"SKILLS_MATCH":<int 0-100>,"EXPERIENCE_RELEVANCE":<int 0-100>,"PROJECT_RELEVANCE":<int 0-100>,"EDUCATION_CERTIFICATION":<int 0-100>,"KEYWORD_DOMAIN_ALIGNMENT":<int 0-100>}},"matched_skills":["<skill>",...],"missing_required_skills":["<skill>",...],"matched_preferred_skills":["<skill>",...],"label_notes":"<2-3 sentences: strongest match point, key gap, and why this score not the adjacent label>","labeled_by":"llm_synthetic","label_version":"rubric_v0.2"}}"""

# ── validation ─────────────────────────────────────────────────────────────────

EMAIL_RE   = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE   = re.compile(r"(?:\+?\d[\s.-]?){8,15}")
URL_RE     = re.compile(r"https?://(?:www\.)?(linkedin|github)\.com/[^\s]+", re.I)


def validate_resume(r: dict, idx: int) -> list[str]:
    errs = []
    for field in ("id", "source", "candidate_level", "raw_text", "summary",
                  "skills", "normalized_skills", "experience_years",
                  "education", "projects", "anonymized", "language"):
        if field not in r:
            errs.append(f"resume line {idx}: missing field '{field}'")
    if r.get("candidate_level") not in ALLOWED_LEVELS:
        errs.append(f"resume {r.get('id')}: invalid candidate_level '{r.get('candidate_level')}'")
    if r.get("anonymized") is not True:
        errs.append(f"resume {r.get('id')}: anonymized must be true")
    if r.get("language") not in ALLOWED_LANGUAGES:
        errs.append(f"resume {r.get('id')}: invalid language '{r.get('language')}'")
    raw = r.get("raw_text", "")
    if EMAIL_RE.search(raw):
        errs.append(f"resume {r.get('id')}: raw_text contains email pattern")
    if URL_RE.search(raw):
        errs.append(f"resume {r.get('id')}: raw_text contains LinkedIn/GitHub URL")
    word_count = len(raw.split())
    if word_count < CV_MIN_WORDS:
        errs.append(
            f"resume {r.get('id')}: raw_text too short ({word_count} words, minimum {CV_MIN_WORDS}). "
            f"Add more job history details, specific technologies, and quantified achievements."
        )
    projects = r.get("projects", [])
    if not projects:
        errs.append(f"resume {r.get('id')}: projects list is empty — must have at least 1 project")
    return errs


def validate_jd(j: dict, idx: int) -> list[str]:
    errs = []
    for field in ("id", "source", "title", "level", "raw_text",
                  "responsibilities", "requirements", "required_skills",
                  "preferred_skills", "min_experience_years", "language"):
        if field not in j:
            errs.append(f"JD line {idx}: missing field '{field}'")
    if j.get("level") not in ALLOWED_LEVELS:
        errs.append(f"JD {j.get('id')}: invalid level '{j.get('level')}'")
    if j.get("language") not in ALLOWED_LANGUAGES:
        errs.append(f"JD {j.get('id')}: invalid language '{j.get('language')}'")
    jd_raw = j.get("raw_text", "")
    word_count = len(jd_raw.split())
    if word_count < JD_MIN_WORDS:
        errs.append(
            f"JD {j.get('id')}: raw_text too short ({word_count} words, minimum {JD_MIN_WORDS}). "
            f"Add team context, responsibilities, tech stack detail, and success criteria."
        )
    required_skills = j.get("required_skills", [])
    if len(required_skills) < 3:
        errs.append(f"JD {j.get('id')}: required_skills has {len(required_skills)} items — minimum 3")
    return errs


def validate_pair(p: dict, resume_ids: set[str], jd_ids: set[str]) -> tuple[list[str], list[str]]:
    errs  = []
    warns = []

    label = p.get("label")
    score = p.get("overall_score")
    pid   = p.get("id", "?")

    for field in ("id", "resume_id", "job_description_id", "label",
                  "overall_score", "criterion_scores",
                  "matched_skills", "missing_required_skills", "label_notes",
                  "labeled_by", "label_version"):
        if field not in p:
            errs.append(f"pair {pid}: missing field '{field}'")

    if label not in ALLOWED_LABELS:
        errs.append(f"pair {pid}: unknown label '{label}' (allowed: {sorted(ALLOWED_LABELS)})")

    if score is None:
        errs.append(f"pair {pid}: missing overall_score")
    else:
        score = int(score)
        if in_boundary_zone(score):
            warns.append(f"pair {pid}: score {score} is in a boundary zone — use valid window instead")
        if label in SCORE_RANGES:
            lo, hi = SCORE_RANGES[label]
            if not (lo <= score <= hi):
                errs.append(f"pair {pid}: score {score} does not match label '{label}' (expected {lo}-{hi})")
        # Check if score is in the valid window for the label (warn only)
        if label in VALID_SCORE_WINDOWS:
            vlo, vhi = VALID_SCORE_WINDOWS[label]
            if not (vlo <= score <= vhi):
                warns.append(
                    f"pair {pid}: score {score} for '{label}' is outside the safe window "
                    f"{vlo}-{vhi} — may be near a boundary zone"
                )

    # Verify weighted score matches criterion scores
    cs = p.get("criterion_scores", {})
    for key in CRITERIA_KEYS:
        if key not in cs:
            errs.append(f"pair {pid}: criterion_scores missing '{key}'")
        elif not isinstance(cs[key], int) or not (0 <= cs[key] <= 100):
            errs.append(f"pair {pid}: criterion_scores.{key} must be int 0-100")

    if cs and score is not None and all(k in cs for k in CRITERIA_KEYS):
        computed = round(
            cs["SKILLS_MATCH"]             * 0.35 +
            cs["EXPERIENCE_RELEVANCE"]     * 0.30 +
            cs["PROJECT_RELEVANCE"]        * 0.15 +
            cs["EDUCATION_CERTIFICATION"]  * 0.10 +
            cs["KEYWORD_DOMAIN_ALIGNMENT"] * 0.10
        )
        if abs(computed - int(score)) > 2:
            errs.append(
                f"pair {pid}: criterion_scores compute to {computed} but overall_score is {score} "
                f"(difference {abs(computed - int(score))} > 2 — scores are inconsistent)"
            )

    notes = p.get("label_notes", "")
    if len(notes.split()) < 10:
        warns.append(f"pair {pid}: label_notes too short ({len(notes.split())} words) — should explain match quality and key gap")

    if p.get("resume_id") not in resume_ids:
        errs.append(f"pair {pid}: resume_id '{p.get('resume_id')}' not found in batch")
    if p.get("job_description_id") not in jd_ids:
        errs.append(f"pair {pid}: job_description_id '{p.get('job_description_id')}' not found in batch")

    if p.get("label_version") != "rubric_v0.2":
        warns.append(f"pair {pid}: label_version is '{p.get('label_version')}', expected 'rubric_v0.2'")

    return errs, warns


def validate_cvjd_ids(
    resumes: list[dict], jds: list[dict],
    resume_start: int, jd_start: int,
) -> list[str]:
    errs: list[str] = []
    expected_resume_ids = {f"resume_{resume_start + i}" for i in range(N_RESUMES)}
    expected_jd_ids     = {f"jd_{jd_start + i}"        for i in range(N_JDS)}

    actual_resume_ids = {r.get("id", f"<missing id on resume #{i}>") for i, r in enumerate(resumes)}
    actual_jd_ids     = {j.get("id", f"<missing id on jd #{i}>")     for i, j in enumerate(jds)}

    for rid in sorted(expected_resume_ids - actual_resume_ids):
        errs.append(f"[ID] resume '{rid}' expected but missing from LLM output")
    for rid in sorted(actual_resume_ids - expected_resume_ids):
        errs.append(
            f"[ID] resume '{rid}' is unexpected "
            f"(expected resume_{resume_start}-resume_{resume_start + N_RESUMES - 1})"
        )
    for jid in sorted(expected_jd_ids - actual_jd_ids):
        errs.append(f"[ID] JD '{jid}' expected but missing from LLM output")
    for jid in sorted(actual_jd_ids - expected_jd_ids):
        errs.append(
            f"[ID] JD '{jid}' is unexpected "
            f"(expected jd_{jd_start}-jd_{jd_start + N_JDS - 1})"
        )
    return errs


def validate_pair_ids(pairs: list[dict], pair_start: int) -> list[str]:
    errs: list[str] = []
    expected_pair_ids = {f"pair_{pair_start + i}" for i in range(N_PAIRS)}

    actual_ids: list[str] = [p.get("id", f"<missing id on line {i + 1}>") for i, p in enumerate(pairs)]

    seen: set[str] = set()
    for pid in actual_ids:
        if pid in seen:
            errs.append(f"[ID] pair '{pid}' is duplicated")
        seen.add(pid)

    for pid in sorted(expected_pair_ids - set(actual_ids)):
        errs.append(f"[ID] pair '{pid}' expected but missing from LLM output")
    for pid in sorted(set(actual_ids) - expected_pair_ids):
        errs.append(
            f"[ID] pair '{pid}' is unexpected "
            f"(expected pair_{pair_start}-pair_{pair_start + N_PAIRS - 1})"
        )
    return errs

# ── actions ────────────────────────────────────────────────────────────────────

def action_generate_prompt() -> None:
    session = load_session()
    if session.get("state") not in ("idle", None):
        print(f"\n  [!] Session is in state '{session['state']}'.")
        confirm = input("      Start a new batch anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            return

    domain = random.choice(DOMAINS)
    resume_start, jd_start, pair_start = get_next_ids()

    label_targets = compute_label_targets(N_PAIRS)
    resume_levels, jd_levels = choose_seniority_mix(label_targets)

    prompt = build_cvjd_prompt(domain, resume_start, jd_start, resume_levels, jd_levels)

    save_session({
        "state":         "waiting_cvjd",
        "domain":        domain,
        "resume_start":  resume_start,
        "jd_start":      jd_start,
        "pair_start":    pair_start,
        "resume_levels": resume_levels,
        "jd_levels":     jd_levels,
    })

    TEMP_INPUT.write_text("", encoding="utf-8")

    label_order = ["poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"]
    print(f"\n  Domain  : {domain}")
    print(f"  IDs     : resume_{resume_start}-{resume_start + N_RESUMES - 1}  |  jd_{jd_start}-{jd_start + N_JDS - 1}")
    print(f"  Pairs   : pair_{pair_start}-{pair_start + N_PAIRS - 1}  ({N_PAIRS} pairs)")
    print(f"  Seniority mix : resumes={resume_levels}  jds={jd_levels}")
    print(f"  Batch targets : " + "  ".join(f"{l}: {label_targets.get(l, 0)}" for l in label_order))
    PROMPT_OUTPUT.write_text(prompt, encoding="utf-8")
    print(f"\n  Prompt written to:  scripts/prompt_output.txt")
    print(f"  → Copy all content, paste into your LLM.")
    print(f"  → Paste LLM output into:  scripts/temp_input.txt")
    print(f"  → Then press [2] to save.")


def action_save() -> None:
    session = load_session()
    state   = session.get("state", "idle")

    if state == "idle":
        print("\n  [!] Nothing to save. Press [1] to generate a prompt first.")
        return

    if not TEMP_INPUT.exists() or not TEMP_INPUT.read_text(encoding="utf-8").strip():
        print(f"\n  [!] {TEMP_INPUT} is empty. Paste the LLM output there first.")
        return

    raw   = TEMP_INPUT.read_text(encoding="utf-8").strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # ── save CV+JD ────────────────────────────────────────────────────────────
    if state == "waiting_cvjd":
        expected_lines = N_RESUMES + N_JDS
        if len(lines) != expected_lines:
            print(f"\n  [!] Expected {expected_lines} lines ({N_RESUMES} resumes + {N_JDS} JDs), got {len(lines)}.")
            print("      Fix the LLM output in temp_input.txt and press [2] again.")
            return

        resumes_new: list[dict] = []
        jds_new:     list[dict] = []
        parse_errors: list[str] = []

        for i, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors.append(f"Line {i}: invalid JSON — {e}")
                continue

            if "candidate_level" in obj:
                resumes_new.append(obj)
            elif "responsibilities" in obj or "required_skills" in obj:
                jds_new.append(obj)
            else:
                parse_errors.append(f"Line {i}: cannot identify as resume or JD")

        if parse_errors:
            print("\n  [!] Parse errors:")
            for e in parse_errors:
                print(f"      ✗ {e}")
            return

        if len(resumes_new) != N_RESUMES or len(jds_new) != N_JDS:
            print(f"\n  [!] Expected {N_RESUMES} resumes + {N_JDS} JDs, got {len(resumes_new)} + {len(jds_new)}.")
            return

        id_errors = validate_cvjd_ids(
            resumes_new, jds_new,
            session["resume_start"], session["jd_start"],
        )
        if id_errors:
            print(f"\n  [!] {len(id_errors)} ID error(s) — NOT saved:")
            for e in id_errors:
                print(f"      ✗ {e}")
            print("\n      Fix the IDs in temp_input.txt and press [2] again.")
            return

        all_errors: list[str] = []
        for i, r in enumerate(resumes_new, 1):
            all_errors.extend(validate_resume(r, i))
        for i, j in enumerate(jds_new, 1):
            all_errors.extend(validate_jd(j, i))

        if all_errors:
            print(f"\n  [!] {len(all_errors)} validation error(s):")
            for e in all_errors:
                print(f"      ✗ {e}")
            print("\n      Fix the output in temp_input.txt and press [2] again.")
            return

        # Print word count summary before saving
        print("\n  Word count check:")
        for r in resumes_new:
            wc = len(r.get("raw_text", "").split())
            print(f"    {r['id']}: {wc} words  {'✓' if wc >= CV_MIN_WORDS else f'✗ (min {CV_MIN_WORDS})'}")
        for j in jds_new:
            wc = len(j.get("raw_text", "").split())
            print(f"    {j['id']}: {wc} words  {'✓' if wc >= JD_MIN_WORDS else f'✗ (min {JD_MIN_WORDS})'}")

        append_jsonl(RESUMES_PATH, resumes_new)
        append_jsonl(JDS_PATH,     jds_new)
        print(f"\n  ✓ Saved {len(resumes_new)} resumes → {RESUMES_PATH.name}")
        print(f"  ✓ Saved {len(jds_new)} JDs      → {JDS_PATH.name}")

        session["state"]   = "waiting_pairs"
        session["resumes"] = resumes_new
        session["jds"]     = jds_new
        save_session(session)

        TEMP_INPUT.write_text("", encoding="utf-8")

        label_targets = compute_label_targets(N_PAIRS)
        current_dist  = dict(Counter(p.get("label") for p in load_jsonl(PAIRS_PATH) if p.get("label") in ALLOWED_LABELS))
        pair_prompt   = build_pair_prompt(resumes_new, jds_new, session["pair_start"], label_targets, current_dist)
        label_order   = ["poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"]
        print(f"  Current distribution: " + "  ".join(f"{l}: {current_dist.get(l, 0)}" for l in label_order))
        print(f"  Batch targets:        " + "  ".join(f"{l}: {label_targets.get(l, 0)}" for l in label_order))
        PROMPT_OUTPUT.write_text(pair_prompt, encoding="utf-8")
        print(f"\n  ✓ Pair prompt written to:  scripts/prompt_output.txt")
        print(f"  → Copy all content, paste into your LLM.")
        print(f"  → Paste LLM output into:  scripts/temp_input.txt")
        print(f"  → Then press [2] to save pairs.")

    # ── save pairs ────────────────────────────────────────────────────────────
    elif state == "waiting_pairs":
        if len(lines) != N_PAIRS:
            print(f"\n  [!] Expected {N_PAIRS} lines ({N_PAIRS} pairs), got {len(lines)}.")
            print("      Fix the output in temp_input.txt and press [2] again.")
            return

        resumes_batch: list[dict] = session.get("resumes", [])
        jds_batch:     list[dict] = session.get("jds",     [])
        resume_ids = {r["id"] for r in resumes_batch}
        jd_ids     = {j["id"] for j in jds_batch}

        pairs_new: list[dict] = []
        all_errors:  list[str] = []
        all_warnings: list[str] = []
        parse_errors: list[str] = []

        for i, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors.append(f"Line {i}: invalid JSON — {e}")
                continue
            errs, warns = validate_pair(obj, resume_ids, jd_ids)
            all_errors.extend(errs)
            all_warnings.extend(warns)
            pairs_new.append(obj)

        if parse_errors:
            print("\n  [!] Parse errors:")
            for e in parse_errors:
                print(f"      ✗ {e}")
            return

        pair_id_errors = validate_pair_ids(pairs_new, session["pair_start"])
        if pair_id_errors:
            print(f"\n  [!] {len(pair_id_errors)} pair ID error(s) — NOT saved:")
            for e in pair_id_errors:
                print(f"      ✗ {e}")
            print("\n      Fix the pair IDs in temp_input.txt and press [2] again.")
            return

        if all_errors:
            print(f"\n  [!] {len(all_errors)} validation error(s) — NOT saved:")
            for e in all_errors:
                print(f"      ✗ {e}")
            print("\n      Fix the output in temp_input.txt and press [2] again.")
            return

        if all_warnings:
            print(f"\n  [!] {len(all_warnings)} warning(s):")
            for w in all_warnings:
                print(f"      ⚠  {w}")
            confirm = input("\n      Save anyway? [y/N]: ").strip().lower()
            if confirm != "y":
                print("      Aborted. Fix boundary scores and press [2] again.")
                return

        append_jsonl(PAIRS_PATH, pairs_new)
        print(f"\n  ✓ Saved {len(pairs_new)} pairs → {PAIRS_PATH.name}")

        dist = Counter(p.get("label") for p in pairs_new)
        print(f"  Batch label breakdown: ", end="")
        print("  |  ".join(f"{k}: {v}" for k, v in dist.items()))

        TEMP_INPUT.write_text("", encoding="utf-8")
        save_session({"state": "idle"})

        show_stats()

# ── main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║      Synthetic CV-JD Data Generator  (v2)        ║")
    print(f"║      Batch: {N_RESUMES} resumes × {N_JDS} JDs = {N_PAIRS} pairs           ║")
    print("╚══════════════════════════════════════════════════╝")
    show_stats()

    while True:
        session = load_session()
        state   = session.get("state", "idle")

        print("\nMenu:")
        print("  [1]  Generate CV+JD prompt  (start new batch)")
        print("  [2]  Save result            (paste LLM output into scripts/temp_input.txt first)")
        print("  [0]  Exit")

        if state != "idle":
            print(f"\n  Current state : {state}")
            print(f"  Domain        : {session.get('domain', '—')}")

        choice = input("\n  Choice: ").strip()

        if choice == "1":
            action_generate_prompt()
        elif choice == "2":
            action_save()
        elif choice == "0":
            print("\n  Bye.\n")
            sys.exit(0)
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
