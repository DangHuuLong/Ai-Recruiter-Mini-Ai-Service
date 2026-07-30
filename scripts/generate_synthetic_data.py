"""
Synthetic data generator for CV-JD pairs (raw_v3).

Workflow — 5 rounds of 2x2 CV/JD generation, then one combined pair-scoring pass:
  [1] Generate CV+JD prompt for the current round → paste into LLM → paste output
      into scripts/temp_input.txt
  [2] Save result → after rounds 1-4: auto-shows the next round's CV+JD prompt
                   → after round 5:   shows one combined pair prompt (20 pairs
                                       across all 5 rounds — rounds are never
                                       cross-paired with each other)
                   → after pairs saved: auto-commits, shows updated stats
  [0] Exit

Run:
  python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPTS_DIR  = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent

RESUMES_PATH = PROJECT_ROOT / "datasets" / "raw_v3" / "resumes.jsonl"
JDS_PATH     = PROJECT_ROOT / "datasets" / "raw_v3" / "job_descriptions.jsonl"
PAIRS_PATH   = PROJECT_ROOT / "datasets" / "raw_v3" / "cv_jd_pairs.jsonl"
TEMP_INPUT       = SCRIPTS_DIR / "temp_input.txt"
PROMPT_OUTPUT    = SCRIPTS_DIR / "prompt_output.txt"
SESSION_FILE     = SCRIPTS_DIR / ".synthetic_session.json"
DOMAIN_USAGE_FILE = SCRIPTS_DIR / ".domain_usage.json"

# ── batch size constants ───────────────────────────────────────────────────────

N_RESUMES         = 2
N_JDS             = 2
N_PAIRS_PER_ROUND = N_RESUMES * N_JDS       # 4
ROUNDS_PER_GROUP  = 5
N_PAIRS           = N_PAIRS_PER_ROUND * ROUNDS_PER_GROUP  # 20 — scored in one pass

# ── constants ─────────────────────────────────────────────────────────────────

ALLOWED_LABELS = {"poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"}

SCORE_RANGES = {
    "poor_match":      (0,  39),
    "weak_match":      (40, 59),
    "moderate_match":  (60, 74),
    "strong_match":    (75, 89),
    "excellent_match": (90, 100),
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
# Expanded vs raw_v2 (18 → 36 entries) specifically to fix a measured problem:
# raw_v2 reused only 18 domain strings ~22x each across ~400 batches, and only
# 16% of its JDs ended up with a unique `domain` tag (vs 53% for the dataset
# that produced the historical 60.76% ceiling). More entries + the rotation in
# pick_domain() below directly target that gap.
DOMAINS = [
    # IT — backend / infra
    "Backend API Development (Python, FastAPI, Node.js, PostgreSQL)",
    "Backend Development for Fintech Payments (Java, Spring Boot, Kafka)",
    "Backend Development for Healthtech (Python, Django, HL7/FHIR integrations)",
    "DevOps and Cloud Infrastructure (AWS, Docker, Kubernetes, Terraform)",
    "Site Reliability Engineering (Prometheus, Grafana, on-call, incident response)",
    "Database Engineering (PostgreSQL, MongoDB, Redis, data modeling)",
    "Platform Engineering for E-commerce (Go, gRPC, high-throughput services)",
    # IT — frontend / mobile / fullstack
    "Frontend Development (React, TypeScript, Next.js, Tailwind CSS)",
    "Frontend Development for Design Systems (React, Storybook, component libraries)",
    "Mobile Development (React Native, Flutter, Swift, Kotlin)",
    "Full-Stack Web Development (Python backend + React frontend)",
    "Full-Stack Development for SaaS Startups (Node.js, React, Stripe billing)",
    # IT — data / AI
    "Data Science and Machine Learning (Python, scikit-learn, PyTorch, pandas)",
    "AI and NLP Engineering (Transformers, LangChain, RAG, embeddings)",
    "Data Engineering and Analytics Pipelines (Airflow, Spark, dbt, BigQuery)",
    "Computer Vision Engineering (OpenCV, PyTorch, object detection, edge deployment)",
    "MLOps and Model Deployment (MLflow, Kubeflow, model serving, monitoring)",
    # IT — security / QA / PM
    "Cybersecurity and Information Security (pentesting, SIEM, network)",
    "Application Security Engineering (SAST/DAST, secure code review, threat modeling)",
    "QA and Test Automation Engineering (Selenium, Playwright, CI test pipelines)",
    "Product Management in Technology (roadmap, agile, stakeholders)",
    "Technical Program Management (cross-team delivery, engineering roadmaps)",
    # Non-IT — business
    "Digital Marketing and Content Strategy (SEO, social media, analytics)",
    "Performance Marketing and Paid Acquisition (Google Ads, Meta Ads, attribution)",
    "Finance and Accounting in Fintech (financial modeling, reporting, compliance)",
    "Investment Analysis and Corporate Finance (valuation, financial modeling, M&A)",
    "Human Resources and Talent Acquisition (recruiting, HRIS, onboarding)",
    "Learning and Development / Corporate Training (curriculum design, LMS)",
    "Supply Chain and Logistics Management (procurement, inventory, ERP)",
    "Retail Operations and Merchandising (planograms, category management)",
    "Business Analysis and Project Management (requirements, BPMN, stakeholders)",
    "Management Consulting (client delivery, frameworks, stakeholder workshops)",
    "E-commerce and Retail Management (marketplace, operations, merchandising)",
    "Customer Success and Account Management (SaaS renewals, onboarding, churn)",
    # Non-IT — design / healthcare
    "UI/UX Design and Product Design (Figma, user research, prototyping)",
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

# ── domain rotation ────────────────────────────────────────────────────────────

def _load_domain_usage() -> dict[str, int]:
    if DOMAIN_USAGE_FILE.exists():
        try:
            return json.loads(DOMAIN_USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_domain_usage(usage: dict[str, int]) -> None:
    DOMAIN_USAGE_FILE.write_text(json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8")


def pick_domain(exclude: set[str] | None = None) -> str:
    """Pick the least-used domain so far (random tie-break among ties), instead
    of plain `random.choice`. Persisted across runs in .domain_usage.json so the
    rotation holds even across separate script invocations — plain random.choice
    with replacement is what let raw_v2 reuse the same 18 domains ~22x each.
    """
    usage = _load_domain_usage()
    exclude = exclude or set()
    candidates = [d for d in DOMAINS if d not in exclude] or DOMAINS
    min_count = min(usage.get(d, 0) for d in candidates)
    least_used = [d for d in candidates if usage.get(d, 0) == min_count]
    chosen = random.choice(least_used)
    usage[chosen] = usage.get(chosen, 0) + 1
    _save_domain_usage(usage)
    return chosen


def pick_domain_pair() -> tuple[str, str]:
    """Two distinct domains for a cross-domain (genuine poor_match) round."""
    domain_a = pick_domain()
    domain_b = pick_domain(exclude={domain_a})
    return domain_a, domain_b

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


def auto_commit_batch() -> None:
    """Stage and commit this group's dataset files after a successful pair save.

    Scoped to the specific files this script writes (not `git add .`) so it never
    sweeps up unrelated in-progress edits elsewhere in the repo — e.g. this script
    being edited, or other uncommitted work in the working tree.
    """
    files = [RESUMES_PATH, JDS_PATH, PAIRS_PATH, TEMP_INPUT, PROMPT_OUTPUT, SESSION_FILE, DOMAIN_USAGE_FILE]
    try:
        subprocess.run(
            ["git", "add", "--", *[str(f) for f in files]],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", "feat: add new job descriptions, resumes and CV-JD pairs"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"\n  ✓ Committed to git")
        else:
            print(f"\n  [!] git commit skipped: {(result.stdout or result.stderr).strip()}")
    except subprocess.CalledProcessError as e:
        print(f"\n  [!] git add failed: {(e.stderr or str(e)).strip()}")
    except FileNotFoundError:
        print(f"\n  [!] git not found on PATH — skipped auto-commit")

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
    print(f"  Batch size: {N_RESUMES} resumes × {N_JDS} JDs = {N_PAIRS_PER_ROUND} pairs/round, "
          f"{ROUNDS_PER_GROUP} rounds scored together = {N_PAIRS} pairs/group")
    print("  Label distribution:")
    for label in labels_order:
        count = dist.get(label, 0)
        pct   = count / total * 100 if total else 0
        bar   = "█" * int(pct / 2.5)
        print(f"    {label:<20} {count:>5}  ({pct:4.1f}%)  {bar}")

    target = max((dist.get(label, 0) for label in ALLOWED_LABELS), default=0)
    total_need = 0
    print(f"\n  Still needed to balance (target = {target} each):")
    for label in labels_order:
        need = max(0, target - dist.get(label, 0))
        total_need += need
        groups = need // N_PAIRS + (1 if need % N_PAIRS else 0)
        print(f"    {label:<20} +{need:>4}  (~{groups:>3} groups)")
    groups_need = total_need // N_PAIRS + (1 if total_need % N_PAIRS else 0)
    print(f"    Total new pairs needed: ~{total_need}  (~{groups_need} groups)")
    print("=" * 60)


def compute_label_targets(n_pairs: int = N_PAIRS) -> dict[str, int]:
    """Return how many of each allowed label to target across a group of pairs.

    Pure integer water-filling — no percentages, no rounding drift. Repeatedly top
    up whichever label(s) currently sit at the minimum count, raising them only as
    far as the next-distinct count above (or splitting the remaining budget evenly,
    one pair at a time, once there isn't enough left for a full step), until the
    n_pairs budget is spent. A label far behind the rest gets the whole budget;
    labels already tied get an even split; nothing is left idle.
    """
    pairs = load_jsonl(PAIRS_PATH)
    dist  = Counter(p.get("label") for p in pairs if p.get("label") in ALLOWED_LABELS)

    labels = sorted(ALLOWED_LABELS)  # stable order, used to break ties deterministically

    counts:  dict[str, int] = {label: dist.get(label, 0) for label in labels}
    targets: dict[str, int] = {label: 0 for label in labels}
    remaining = n_pairs

    while remaining > 0:
        min_count = min(counts.values())
        lowest    = [l for l in labels if counts[l] == min_count]

        if remaining <= len(lowest):
            for label in lowest[:remaining]:
                counts[label]  += 1
                targets[label] += 1
            break

        higher = [c for c in counts.values() if c > min_count]
        step   = (min(higher) - min_count) if higher else remaining // len(lowest)
        step   = max(1, min(step, remaining // len(lowest)))

        for label in lowest:
            counts[label]  += step
            targets[label] += step
        remaining -= step * len(lowest)

    return targets


LEVELS = ["junior", "middle", "senior"]

# Multiple phrasings per tier — chosen randomly per round so the CV/JD generation
# prompt doesn't repeat the exact same sentence hundreds of times. raw_v2 reused
# one fixed string per tier verbatim across ~400 batches, and the same handful of
# opening sentences ("Over the past four years, I...") showed up 27x as a result.
LEVEL_INFO_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "junior": [
        ("0-1 yr, fresh graduate or first job",              "min 1 yr, entry/associate level role"),
        ("0-1 yr, recent graduate entering the field",        "min 1 yr, junior/associate opening"),
        ("under 1 yr, first professional role after school",  "min 1 yr, entry-level position"),
    ],
    "middle": [
        ("3-5 yrs total experience",                          "min 3 yrs, mid-level individual contributor"),
        ("3-5 yrs, established individual contributor",       "min 3 yrs, experienced individual contributor role"),
        ("4-5 yrs of hands-on professional experience",       "min 3 yrs, mid-level opening"),
    ],
    "senior": [
        ("6-9 yrs, tech lead or principal",                   "min 6 yrs, lead or senior engineer, deep specialization required"),
        ("7-9 yrs, senior specialist or team lead",           "min 6 yrs, senior-level position requiring deep expertise"),
        ("6-8 yrs, principal-track professional",             "min 6 yrs, senior/lead opening with high ownership"),
    ],
}


def pick_level_info(level: str) -> tuple[str, str]:
    return random.choice(LEVEL_INFO_VARIANTS[level])


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
    n_resumes: int = N_RESUMES,
    n_jds: int = N_JDS,
) -> tuple[list[str], list[str]]:
    """Pick how many junior/middle/senior resumes and JDs to generate this round.

    For every candidate composition, compute what the WHOLE dataset's label
    counts would look like after adding this round, and pick whichever leaves
    that resulting distribution most level — i.e. the smallest resulting
    maximum, then the smallest 2nd-largest, and so on (comparing the resulting
    counts sorted descending, lexicographically). Plain greedy load-balancing
    computed directly on real outcomes.
    """
    dist_now   = Counter(p.get("label") for p in load_jsonl(PAIRS_PATH) if p.get("label") in ALLOWED_LABELS)
    counts_now = {l: dist_now.get(l, 0) for l in ALLOWED_LABELS}

    best_mix = None
    best_key = None
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
            resulting = {l: counts_now[l] + dist.get(l, 0) for l in ALLOWED_LABELS}
            key = tuple(sorted(resulting.values(), reverse=True))
            if best_key is None or key < best_key:
                best_key = key
                best_mix = (r_counts, j_counts)

    r_counts, j_counts = best_mix
    resume_levels = [lvl for lvl, count in zip(LEVELS, r_counts) for _ in range(count)]
    jd_levels     = [lvl for lvl, count in zip(LEVELS, j_counts) for _ in range(count)]
    return resume_levels, jd_levels


def choose_round_spec(n_resumes: int = N_RESUMES, n_jds: int = N_JDS) -> dict:
    """Decide one round's generation mode: cross_domain or same_domain.

    If poor_match is currently the SINGLE most-needed label (not tied), dedicate
    this round to a genuine cross-domain pairing (resume domain != JD domain)
    instead of the same-domain seniority-gap mechanism. Real poor matches in
    production come from mismatched fields, not same-field candidates who are
    merely junior — and this also raises overall domain diversity, which was
    measured to be the main driver of raw_v2's overfitting (see
    docs/similarity-model-cross-encoder-v0.7.md, Section 6.3).
    """
    dist_now = Counter(p.get("label") for p in load_jsonl(PAIRS_PATH) if p.get("label") in ALLOWED_LABELS)
    counts   = {l: dist_now.get(l, 0) for l in ALLOWED_LABELS}
    min_count = min(counts.values())
    lowest    = [l for l in ALLOWED_LABELS if counts[l] == min_count]

    if lowest == ["poor_match"]:
        resume_domain, jd_domain = pick_domain_pair()
        return {
            "mode":          "cross_domain",
            "resume_domain": resume_domain,
            "jd_domain":     jd_domain,
            "resume_levels": [random.choice(LEVELS) for _ in range(n_resumes)],
            "jd_levels":     [random.choice(LEVELS) for _ in range(n_jds)],
        }

    resume_levels, jd_levels = choose_seniority_mix(n_resumes, n_jds)
    return {
        "mode":          "same_domain",
        "domain":        pick_domain(),
        "resume_levels": resume_levels,
        "jd_levels":     jd_levels,
    }

# ── next IDs ───────────────────────────────────────────────────────────────────

def get_next_ids() -> tuple[int, int, int]:
    resume_start = get_max_numeric_id(load_jsonl(RESUMES_PATH), "resume_") + 1
    jd_start     = get_max_numeric_id(load_jsonl(JDS_PATH),     "jd_")     + 1
    pair_start   = get_max_numeric_id(load_jsonl(PAIRS_PATH),   "pair_")   + 1
    return resume_start, jd_start, pair_start

# ── prompt builders ────────────────────────────────────────────────────────────

def build_cvjd_prompt(spec: dict, resume_start: int, jd_start: int) -> str:
    r_ids  = [f"resume_{resume_start + i}" for i in range(N_RESUMES)]
    j_ids  = [f"jd_{jd_start + i}"        for i in range(N_JDS)]

    resume_levels = spec["resume_levels"]
    jd_levels     = spec["jd_levels"]

    resume_lines = "\n".join(
        f"Resume {rid}: {lvl:<8} ({pick_level_info(lvl)[0]})"
        for rid, lvl in zip(r_ids, resume_levels)
    )
    jd_lines = "\n".join(
        f"JD {jid}: {lvl:<8} ({pick_level_info(lvl)[1]})"
        for jid, lvl in zip(j_ids, jd_levels)
    )

    if spec["mode"] == "cross_domain":
        domain_header = f"""Generate exactly {N_RESUMES} resumes in the domain: {spec['resume_domain']}
Generate exactly {N_JDS} job descriptions in an UNRELATED domain: {spec['jd_domain']}

These resumes and these job descriptions are DELIBERATELY from different fields.
Every cross-pair in this round is a genuine poor_match — a candidate applying
completely outside their field — not a same-field candidate who merely lacks
experience."""
        expected_dist: Counter = Counter({"poor_match": N_RESUMES * N_JDS})
        skill_guidance = """════════════════════════════════════
SKILL AUTHORING — CROSS-DOMAIN, NO FORCED OVERLAP
════════════════════════════════════
Write each resume's skills authentically for ITS OWN domain, and each JD's
required_skills authentically for ITS OWN (different) domain. Do not engineer
any skill overlap between them — a real candidate from one field naturally has
almost nothing in common with a job posting from an unrelated field. That gap
is the entire point of this round; do not soften it."""
    else:
        domain_header = f"Generate exactly {N_RESUMES} resumes and {N_JDS} job descriptions in the domain: {spec['domain']}"
        expected_dist = Counter()
        for r_lvl in resume_levels:
            for j_lvl in jd_levels:
                gap = LEVELS.index(j_lvl) - LEVELS.index(r_lvl)
                expected_dist[natural_label_for_gap(gap)] += 1
        skill_guidance = """════════════════════════════════════
SKILL AUTHORING — USE JUDGMENT, NOT A FIXED PERCENTAGE
════════════════════════════════════
Design each resume's skill list based on genuine expertise for its stated
seniority — do NOT target an exact overlap percentage with any specific JD.
A junior resume should naturally know fewer of a senior JD's specialized or
advanced-tier tools (they simply haven't needed them yet), while a resume at
the same level as a JD should naturally share most of its core stack. Let the
overlap emerge from realistic, individually-authored skill sets, not a formula."""

    expected_lines = "\n".join(f"  {label:<16} × {count}" for label, count in expected_dist.items())

    return f"""You are a dataset generator for an AI recruitment system.
{domain_header}

These {N_RESUMES + N_JDS} documents will be cross-paired into {N_PAIRS_PER_ROUND} CV-JD pairs.
Target score range: 0-100 (continuous — do not avoid any particular sub-range).

════════════════════════════════════
SENIORITY ASSIGNMENT (follow exactly)
════════════════════════════════════
{resume_lines}

{jd_lines}

This round's mix was chosen (based on which labels the dataset currently lacks)
so that cross-pairing every resume × every JD naturally produces this label
spread — score the resulting pairs honestly instead of forcing a different
distribution:
{expected_lines}

Gap rule (JD seniority tier − resume seniority tier), for reference:
  +2 → poor_match       +1 → weak_match
   0 → excellent_match  -1 → strong_match
  <=-2 → moderate_match (heavily overqualified)

{skill_guidance}

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
Vary sentence structure and opening phrasing from any previous batches you may
have generated in this session — do not reuse the same opening sentence pattern.

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


def _resume_summary(r: dict) -> str:
    raw_words = len(r.get("raw_text", "").split())
    return (
        f"  id: {r['id']} | level: {r.get('candidate_level')} | "
        f"exp: {r.get('experience_years')} yrs | raw_text: {raw_words} words\n"
        f"  skills: {', '.join(r.get('skills', []))}\n"
        f"  projects: {', '.join(p.get('name','') for p in r.get('projects',[]))}\n"
        f"  background: {r.get('raw_text', '')[:300]}"
    )


def _jd_summary(j: dict) -> str:
    raw_words = len(j.get("raw_text", "").split())
    return (
        f"  id: {j['id']} | title: {j.get('title')} | level: {j.get('level')} | "
        f"min_exp: {j.get('min_experience_years')} yrs | raw_text: {raw_words} words\n"
        f"  required_skills: {', '.join(j.get('required_skills', []))}\n"
        f"  preferred_skills: {', '.join(j.get('preferred_skills', []))}\n"
        f"  description: {j.get('raw_text', '')[:250]}"
    )


def build_group_pair_prompt(
    rounds: list[dict],
    label_targets: dict[str, int] | None = None,
    current_dist: dict[str, int] | None = None,
    only_pair_ids: set[str] | None = None,
) -> str:
    """Combine ROUNDS_PER_GROUP rounds' worth of resumes/JDs into ONE scoring
    prompt. Pairing stays within each round only — rounds are never cross-paired
    with each other, so the total is exactly len(rounds) * N_PAIRS_PER_ROUND.

    `only_pair_ids`, if given, restricts the prompt to just those pair ids —
    used for incremental retries after a response came back incomplete, so
    the retry asks for (and the model has to output) only what's still
    missing instead of regenerating the whole group and risking the same
    truncation again.
    """
    def round_pair_ids(rnd: dict) -> list[str]:
        return [f"pair_{rnd['pair_start'] + i}" for i in range(N_PAIRS_PER_ROUND)]

    if only_pair_ids is not None:
        rounds = [rnd for rnd in rounds if set(round_pair_ids(rnd)) & only_pair_ids]

    all_resumes = [r for rnd in rounds for r in rnd["resumes"]]
    all_jds     = [j for rnd in rounds for j in rnd["jds"]]

    resume_block = "\n\n".join(_resume_summary(r) for r in all_resumes)
    jd_block     = "\n\n".join(_jd_summary(j)     for j in all_jds)

    pairing_blocks = []
    n_to_score = 0
    for rnd in rounds:
        r_ids = [r["id"] for r in rnd["resumes"]]
        j_ids = [j["id"] for j in rnd["jds"]]
        spec  = rnd["spec"]

        if spec["mode"] == "cross_domain":
            header = f"Round {rnd['index']} (CROSS-DOMAIN: {spec['resume_domain']} × {spec['jd_domain']} — expect poor_match throughout):"
        else:
            header = f"Round {rnd['index']} (domain: {spec['domain']}):"

        lines = [header]
        idx = rnd["pair_start"]
        wrote_any = False
        for rid in r_ids:
            for jid in j_ids:
                pair_id = f"pair_{idx}"
                idx += 1
                if only_pair_ids is not None and pair_id not in only_pair_ids:
                    continue
                r_lvl = next(r.get("candidate_level") for r in rnd["resumes"] if r["id"] == rid)
                j_lvl = next(j.get("level") for j in rnd["jds"] if j["id"] == jid)
                if spec["mode"] == "cross_domain":
                    expected = "poor_match"
                elif r_lvl in LEVELS and j_lvl in LEVELS:
                    expected = natural_label_for_gap(LEVELS.index(j_lvl) - LEVELS.index(r_lvl))
                else:
                    expected = "?"
                lines.append(f"  {pair_id:<10}  {rid} × {jid}   (expected: {expected})")
                wrote_any = True
                n_to_score += 1
        if wrote_any:
            pairing_blocks.append("\n".join(lines))
    pair_order = "\n\n".join(pairing_blocks)

    partial_note = ""
    if only_pair_ids is not None:
        partial_note = f"""
════════════════════════════════════
PARTIAL RE-REQUEST
════════════════════════════════════
A previous response was incomplete. Score ONLY the {n_to_score} pairs listed
below — do not re-score or repeat any other pair.
"""

    label_order = ["poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"]

    expected_dist: Counter = Counter()
    for rnd in rounds:
        spec = rnd["spec"]
        if spec["mode"] == "cross_domain":
            expected_dist["poor_match"] += len(rnd["resumes"]) * len(rnd["jds"])
        else:
            for r in rnd["resumes"]:
                for j in rnd["jds"]:
                    r_lvl, j_lvl = r.get("candidate_level"), j.get("level")
                    if r_lvl in LEVELS and j_lvl in LEVELS:
                        expected_dist[natural_label_for_gap(LEVELS.index(j_lvl) - LEVELS.index(r_lvl))] += 1

    if label_targets or expected_dist:
        total_existing = sum((current_dist or {}).values())
        rows = []
        for label in label_order:
            existing = (current_dist or {}).get(label, 0)
            expected = expected_dist.get(label, 0)
            rows.append(f"  {label:<20}  existing: {existing:>4}  →  expected this group: {expected} pairs (from rounds below)")
        dist_lines = "\n".join(rows)
        distribution_section = f"""
════════════════════════════════════
LABEL BALANCE — CONTEXT (already engineered into this group)
════════════════════════════════════
Overall dataset so far: {total_existing} pairs total.
This group's resume/JD seniority levels and domains were deliberately chosen so
that scoring each pair honestly (per the rubric below) should already land close
to this spread — no need to force it, just verify your scores roughly match:

{dist_lines}

Rules:
• Score every pair from its actual documents — seniority gap, skill overlap, project fit.
• If a pair's honest score lands far from its "expected" label above, trust the
  documents over the expectation and note why in label_notes.
• Never pair a resume with a JD from a different round.
"""
    else:
        distribution_section = ""

    return f"""You are a dataset labeler for an AI recruitment system.
Score {n_to_score} CV-JD pairs (from {len(rounds)} round(s)) using rubric v0.2.
{partial_note}
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

Scores are CONTINUOUS across the full 0-100 range — including near a boundary
(e.g. 58, 61, 74, 76). Do not avoid any sub-range. A genuinely borderline pair
should get a genuinely borderline score; do not round it away from the boundary
just to make the label look cleaner.

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
A valid poor_match pair (score 0-39) requires ALL of:
  1. Massive experience gap OR domain mismatch (cross-domain rounds: domain
     mismatch alone is sufficient regardless of experience_years)
     → EXPERIENCE_RELEVANCE: 0-30
  2. Core skill mismatch: matched/required skill ratio < 40%
     → SKILLS_MATCH: 0-35 (must equal the computed ratio, not a template number)
  3. Project misalignment: no relevant projects that map to JD responsibilities
     → PROJECT_RELEVANCE: 0-30
Natural sources: (a) intern/junior CV × senior JD in the SAME domain, or
(b) any-seniority CV × any-seniority JD in a CROSS-DOMAIN round (see rounds above).

DECISION PRIORITY — resolve same-domain intern×senior pairs with this BEFORE
picking a label: if resume experience_years <= 1 AND JD min_experience_years >= 6,
compute the skill overlap ratio (matched required_skills / total required_skills) FIRST:
  • ratio < 40%  → this pair MUST be poor_match. The weak_match "experience gap"
    rule below does NOT apply to interns (<=1 yr) — it only applies once the
    candidate has 1-2+ yrs and is past entry level.
  • ratio >= 40% → this pair is weak_match instead (real partial overlap saves it).
This ratio check overrides any temptation to soften the label because the CV and
JD share some domain vocabulary — shared domain alone is not overlap.
For cross-domain rounds, skip this check entirely — domain mismatch alone
determines poor_match.

════════════════════════════════════
weak_match SCORING GUIDANCE
════════════════════════════════════
A valid weak_match pair (score 40-59) requires at least one of:
  1. Experience gap: resume has >1 yr (already past intern/fresh-grad stage — e.g.
     1-2 yrs) vs senior JD (5+ yrs required). Do NOT use this rule for resumes with
     experience_years <= 1 — see the poor_match DECISION PRIORITY above, which
     routes those to poor_match unless overlap ratio is already >= 40%.
     → EXPERIENCE_RELEVANCE: 20-45 (large gap penalised heavily)
     → SKILLS_MATCH: 40-60 (must equal the actual computed overlap ratio, not a default)
  2. Skill domain mismatch: same broad field but different specialization
     → SKILLS_MATCH: 35-55 (missing core required stack)
     → PROJECT_RELEVANCE: 30-55 (projects don't align with JD responsibilities)

For ALL pairs, criterion scores must be INTERNALLY CONSISTENT:
  • If EXPERIENCE_RELEVANCE is 20, explain in label_notes (e.g. "requires 6+ yrs, candidate has 0.5 yrs")
  • SKILLS_MATCH cannot be 85 if overall is 28 — the math must work
  • Verify: round(SM×0.35 + ER×0.30 + PR×0.15 + EC×0.10 + KD×0.10) == overall_score
{distribution_section}
════════════════════════════════════
PAIRING ORDER ({n_to_score} pairs across {len(rounds)} round(s) — never cross-pair rounds)
════════════════════════════════════
{pair_order}

════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════
Output exactly {n_to_score} lines.
Each line is a single JSON object (no array brackets, no commas between lines).
No markdown, no explanation — only {n_to_score} JSON lines.

JSON schema per pair:
{{"id":"<pair_id>","resume_id":"<resume_id>","job_description_id":"<jd_id>","split":null,"label":"<poor_match|weak_match|moderate_match|strong_match|excellent_match>","overall_score":<int 0-100>,"criterion_scores":{{"SKILLS_MATCH":<int 0-100>,"EXPERIENCE_RELEVANCE":<int 0-100>,"PROJECT_RELEVANCE":<int 0-100>,"EDUCATION_CERTIFICATION":<int 0-100>,"KEYWORD_DOMAIN_ALIGNMENT":<int 0-100>}},"matched_skills":["<skill>",...],"missing_required_skills":["<skill>",...],"matched_preferred_skills":["<skill>",...],"label_notes":"<2-3 sentences: strongest match point, key gap, and why this score not the adjacent label>","labeled_by":"llm_synthetic","label_version":"rubric_v0.2"}}"""

# ── validation ─────────────────────────────────────────────────────────────────

EMAIL_RE   = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
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
        if label in SCORE_RANGES:
            lo, hi = SCORE_RANGES[label]
            if not (lo <= score <= hi):
                errs.append(f"pair {pid}: score {score} does not match label '{label}' (expected {lo}-{hi})")

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
        errs.append(f"pair {pid}: resume_id '{p.get('resume_id')}' not found in this group")
    if p.get("job_description_id") not in jd_ids:
        errs.append(f"pair {pid}: job_description_id '{p.get('job_description_id')}' not found in this group")

    if p.get("label_version") != "rubric_v0.2":
        warns.append(f"pair {pid}: label_version is '{p.get('label_version')}', expected 'rubric_v0.2'")

    return errs, warns


def validate_pair_round_consistency(pairs: list[dict], rounds: list[dict]) -> list[str]:
    """A pair's resume and JD must come from the SAME round — rounds are never
    cross-paired when building the group prompt, so any pair claiming otherwise
    is a labeling mistake, not a legitimate cross-round pair."""
    resume_round_of: dict[str, int] = {}
    jd_round_of:     dict[str, int] = {}
    for rnd in rounds:
        for r in rnd["resumes"]:
            resume_round_of[r["id"]] = rnd["index"]
        for j in rnd["jds"]:
            jd_round_of[j["id"]] = rnd["index"]

    errs = []
    for p in pairs:
        r_round = resume_round_of.get(p.get("resume_id"))
        j_round = jd_round_of.get(p.get("job_description_id"))
        if r_round is not None and j_round is not None and r_round != j_round:
            errs.append(
                f"pair {p.get('id')}: resume is from round {r_round} but JD is from round {j_round} "
                f"— resumes and JDs must only be paired within the same round"
            )
    return errs


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


# ── actions ────────────────────────────────────────────────────────────────────

def _build_round(round_index: int, group_pair_start: int) -> tuple[str, dict]:
    """Build the CV/JD prompt for `round_index` (1-based) and return
    (prompt, round_meta) — round_meta holds everything needed later to build
    the group pair prompt and to save this round's resumes/jds once pasted."""
    resume_start, jd_start, _ = get_next_ids()
    round_pair_start = group_pair_start + (round_index - 1) * N_PAIRS_PER_ROUND

    spec   = choose_round_spec(N_RESUMES, N_JDS)
    prompt = build_cvjd_prompt(spec, resume_start, jd_start)

    round_meta = {
        "index":        round_index,
        "spec":         spec,
        "resume_start": resume_start,
        "jd_start":     jd_start,
        "pair_start":   round_pair_start,
    }
    return prompt, round_meta


def _print_round_prompt(round_meta: dict, prompt: str) -> None:
    spec = round_meta["spec"]
    domain_desc = (
        f"{spec['resume_domain']} × {spec['jd_domain']} (CROSS-DOMAIN)"
        if spec["mode"] == "cross_domain" else spec["domain"]
    )
    print(f"\n  Round   : {round_meta['index']}/{ROUNDS_PER_GROUP}")
    print(f"  Domain  : {domain_desc}")
    print(f"  IDs     : resume_{round_meta['resume_start']}-{round_meta['resume_start'] + N_RESUMES - 1}  "
          f"|  jd_{round_meta['jd_start']}-{round_meta['jd_start'] + N_JDS - 1}")
    print(f"  Seniority mix : resumes={spec['resume_levels']}  jds={spec['jd_levels']}")
    PROMPT_OUTPUT.write_text(prompt, encoding="utf-8")
    print(f"\n  Prompt written to:  scripts/prompt_output.txt")
    print(f"  → Copy all content, paste into your LLM.")
    print(f"  → Paste LLM output into:  scripts/temp_input.txt")
    print(f"  → Then press [2] to save.")


def _process_cvjd_lines(lines: list[str], round_meta: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Parse + validate one round's CV/JD lines. Pure — no printing, no I/O
    besides what's already in `lines`. Shared by the interactive action_save()
    and the fully-automated API path so both enforce identical rules."""
    expected_lines = N_RESUMES + N_JDS
    if len(lines) != expected_lines:
        return [], [], [f"Expected {expected_lines} lines ({N_RESUMES} resumes + {N_JDS} JDs), got {len(lines)}."]

    resumes_new: list[dict] = []
    jds_new:     list[dict] = []
    errors:      list[str]  = []

    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: invalid JSON — {e}")
            continue
        if "candidate_level" in obj:
            resumes_new.append(obj)
        elif "responsibilities" in obj or "required_skills" in obj:
            jds_new.append(obj)
        else:
            errors.append(f"Line {i}: cannot identify as resume or JD")

    if errors:
        return resumes_new, jds_new, errors

    if len(resumes_new) != N_RESUMES or len(jds_new) != N_JDS:
        return resumes_new, jds_new, [f"Expected {N_RESUMES} resumes + {N_JDS} JDs, got {len(resumes_new)} + {len(jds_new)}."]

    errors.extend(validate_cvjd_ids(resumes_new, jds_new, round_meta["resume_start"], round_meta["jd_start"]))
    if errors:
        return resumes_new, jds_new, errors

    for i, r in enumerate(resumes_new, 1):
        errors.extend(validate_resume(r, i))
    for i, j in enumerate(jds_new, 1):
        errors.extend(validate_jd(j, i))

    return resumes_new, jds_new, errors


def _process_pair_lines(
    lines: list[str],
    rounds: list[dict],
    expected_ids: set[str],
    resume_ids: set[str],
    jd_ids: set[str],
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Parse + validate pair-scoring lines against `expected_ids`. Pure —
    shared by the interactive save path (expected_ids = the whole group) and
    the automated incremental path (expected_ids = only the pairs still
    missing from earlier attempts in this group — see _auto_generate_pairs).

    Returns (accepted, errors, warnings): `accepted` maps pair_id -> validated
    pair object for every expected id that parsed AND passed validation in
    THIS response. An id that's missing, or present but fails validation, is
    left out of `accepted` and reported in `errors` instead — a caller that
    needs one-shot completeness (manual save) should treat any errors as
    fatal, while an accumulating caller can just retry the still-missing ids
    without throwing away what already validated correctly.

    Also tolerant of extra/duplicate/garbage lines: LLMs occasionally emit a
    stray or repeated JSON object alongside the expected ones (seen e.g. as
    "got 21" instead of 20) — those are dropped with a warning rather than
    failing everything else in the response.
    """
    parse_errors: list[str] = []
    parsed: list[dict] = []
    for i, line in enumerate(lines, 1):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as e:
            parse_errors.append(f"Line {i}: invalid JSON — {e}")

    warnings: list[str] = []
    by_id: dict[str, dict] = {}
    for obj in parsed:
        pid = obj.get("id")
        if pid not in expected_ids:
            warnings.append(f"[ID] pair '{pid}' is unexpected/extra — dropped")
            continue
        if pid in by_id:
            warnings.append(f"[ID] pair '{pid}' is duplicated — keeping first occurrence")
            continue
        by_id[pid] = obj

    errors: list[str] = list(parse_errors)
    accepted: dict[str, dict] = {}
    for pid, p in by_id.items():
        errs, warns = validate_pair(p, resume_ids, jd_ids)
        errs.extend(validate_pair_round_consistency([p], rounds))
        if errs:
            errors.extend(errs)
            continue
        warnings.extend(warns)
        accepted[pid] = p

    for pid in sorted(expected_ids - set(by_id)):
        errors.append(f"[ID] pair '{pid}' expected but missing from LLM output")

    return accepted, errors, warnings


def action_generate_prompt() -> None:
    session = load_session()
    if session.get("state") not in ("idle", None):
        print(f"\n  [!] Session is in state '{session['state']}'.")
        confirm = input("      Start a new batch anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            return

    _, _, group_pair_start = get_next_ids()
    prompt, round_meta = _build_round(1, group_pair_start)

    save_session({
        "state":            "waiting_cvjd",
        "group_pair_start": group_pair_start,
        "current_round":    round_meta,
        "rounds":           [],
    })

    TEMP_INPUT.write_text("", encoding="utf-8")
    _print_round_prompt(round_meta, prompt)


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

    # ── save one round's CV+JD ──────────────────────────────────────────────
    if state == "waiting_cvjd":
        current_round = session["current_round"]
        resumes_new, jds_new, all_errors = _process_cvjd_lines(lines, current_round)

        if all_errors:
            print(f"\n  [!] {len(all_errors)} error(s):")
            for e in all_errors:
                print(f"      ✗ {e}")
            print("\n      Fix the output in temp_input.txt and press [2] again.")
            return

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

        completed_round = dict(current_round)
        completed_round["resumes"] = resumes_new
        completed_round["jds"]     = jds_new
        rounds = session.get("rounds", []) + [completed_round]

        TEMP_INPUT.write_text("", encoding="utf-8")

        print(f"\n  ✓ Round {completed_round['index']}/{ROUNDS_PER_GROUP} saved.")

        if len(rounds) < ROUNDS_PER_GROUP:
            next_index = len(rounds) + 1
            prompt, next_round_meta = _build_round(next_index, session["group_pair_start"])
            save_session({
                "state":            "waiting_cvjd",
                "group_pair_start": session["group_pair_start"],
                "current_round":    next_round_meta,
                "rounds":           rounds,
            })
            _print_round_prompt(next_round_meta, prompt)
        else:
            label_targets = compute_label_targets(N_PAIRS)
            current_dist  = dict(Counter(p.get("label") for p in load_jsonl(PAIRS_PATH) if p.get("label") in ALLOWED_LABELS))
            pair_prompt   = build_group_pair_prompt(rounds, label_targets, current_dist)

            save_session({
                "state":            "waiting_pairs",
                "group_pair_start": session["group_pair_start"],
                "rounds":           rounds,
            })

            label_order = ["poor_match", "weak_match", "moderate_match", "strong_match", "excellent_match"]
            print(f"\n  All {ROUNDS_PER_GROUP} rounds complete — {N_PAIRS} pairs ready to score.")
            print(f"  Current distribution: " + "  ".join(f"{l}: {current_dist.get(l, 0)}" for l in label_order))
            print(f"  Group targets:        " + "  ".join(f"{l}: {label_targets.get(l, 0)}" for l in label_order))
            PROMPT_OUTPUT.write_text(pair_prompt, encoding="utf-8")
            print(f"\n  ✓ Pair prompt written to:  scripts/prompt_output.txt")
            print(f"  → Copy all content, paste into your LLM.")
            print(f"  → Paste LLM output into:  scripts/temp_input.txt")
            print(f"  → Then press [2] to save pairs.")

    # ── save pairs for the whole group ──────────────────────────────────────
    elif state == "waiting_pairs":
        rounds = session.get("rounds", [])
        resume_ids = {r["id"] for rnd in rounds for r in rnd["resumes"]}
        jd_ids     = {j["id"] for rnd in rounds for j in rnd["jds"]}
        expected_ids = {f"pair_{session['group_pair_start'] + i}" for i in range(N_PAIRS)}

        accepted, all_errors, all_warnings = _process_pair_lines(
            lines, rounds, expected_ids, resume_ids, jd_ids,
        )

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
                print("      Aborted. Fix the output and press [2] again.")
                return

        pairs_new = [accepted[pid] for pid in sorted(expected_ids, key=lambda s: int(s.split("_")[1]))]
        append_jsonl(PAIRS_PATH, pairs_new)
        print(f"\n  ✓ Saved {len(pairs_new)} pairs → {PAIRS_PATH.name}")

        dist = Counter(p.get("label") for p in pairs_new)
        print(f"  Group label breakdown: ", end="")
        print("  |  ".join(f"{k}: {v}" for k, v in dist.items()))

        TEMP_INPUT.write_text("", encoding="utf-8")
        save_session({"state": "idle"})

        auto_commit_batch()
        show_stats()

# ── fully-automated mode (calls .env API keys directly, no copy-paste) ────────

try:
    import llm_client
    _LLM_CLIENT_AVAILABLE = True
except ImportError:
    _LLM_CLIENT_AVAILABLE = False

MAX_ATTEMPTS_PER_STEP = 12  # enough headroom to reach every configured provider,
                             # not just cycle within the first one or two

# How many consecutive validation failures from the SAME provider (across
# different keys) before that provider's remaining keys get skipped too.
PROVIDER_STRIKES_BEFORE_SKIP = 2


def _record_provider_failure(used_key: str, provider_fails: dict[str, int], tried: set[str]) -> None:
    """Track a validation failure from `used_key`'s provider.

    A single bad generation is usually just an unlucky sample (temperature=0.9
    means the same model can produce a fine response on the next call with a
    different key) — so don't give up on a whole provider after one miss.
    Only once the SAME provider has failed validation
    PROVIDER_STRIKES_BEFORE_SKIP times in a row (a real signal that its model
    has a systematic problem with this prompt, e.g. reliably emitting one
    extra line) do we stop wasting attempts on its remaining keys.
    """
    provider = used_key.split(":", 1)[0]
    provider_fails[provider] = provider_fails.get(provider, 0) + 1
    if provider_fails[provider] >= PROVIDER_STRIKES_BEFORE_SKIP:
        tried |= llm_client.provider_tokens(used_key)


def _auto_generate_round(round_index: int, group_pair_start: int) -> tuple[list[dict], list[dict], dict]:
    """Build one round's prompt, call the LLM (with automatic key fallback),
    validate, and on failure retry with a DIFFERENT key — up to
    MAX_ATTEMPTS_PER_STEP times — before giving up on this round. Reuses the
    same prompt across retries; only the API call is repeated."""
    resume_start, jd_start, _ = get_next_ids()
    round_pair_start = group_pair_start + (round_index - 1) * N_PAIRS_PER_ROUND
    spec = choose_round_spec()
    round_meta = {
        "index":        round_index,
        "spec":         spec,
        "resume_start": resume_start,
        "jd_start":     jd_start,
        "pair_start":   round_pair_start,
    }
    prompt = build_cvjd_prompt(spec, resume_start, jd_start)
    PROMPT_OUTPUT.write_text(prompt, encoding="utf-8")

    tried: set[str] = set()
    provider_fails: dict[str, int] = {}
    for attempt in range(1, MAX_ATTEMPTS_PER_STEP + 1):
        raw, used_key = llm_client.call_llm(prompt, exclude=tried)
        tried.add(used_key)
        TEMP_INPUT.write_text(raw, encoding="utf-8")
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        resumes_new, jds_new, errors = _process_cvjd_lines(lines, round_meta)
        if not errors:
            for r in resumes_new:
                wc = len(r.get("raw_text", "").split())
                print(f"    {r['id']}: {wc} words  {'ok' if wc >= CV_MIN_WORDS else 'SHORT'}  (via {used_key})")
            return resumes_new, jds_new, round_meta
        print(f"  [auto] round {round_index} attempt {attempt}/{MAX_ATTEMPTS_PER_STEP} via {used_key} failed validation:")
        for e in errors[:5]:
            print(f"      ✗ {e}")
        _record_provider_failure(used_key, provider_fails, tried)

    raise RuntimeError(f"Round {round_index}: failed validation after {MAX_ATTEMPTS_PER_STEP} attempts across different keys.")


def _auto_generate_pairs(
    rounds: list[dict], label_targets: dict, current_dist: dict, group_pair_start: int,
) -> list[dict]:
    """Score all N_PAIRS pairs, accumulating across retries: each attempt only
    asks for whatever is still missing (a smaller prompt + smaller expected
    response than re-requesting the full group every time), and validated
    pairs from a partial response are kept rather than discarded. This avoids
    wasting an otherwise-good 18/20 response just because the last couple of
    pairs got truncated or mis-scored."""
    expected_ids = {f"pair_{group_pair_start + i}" for i in range(N_PAIRS)}
    resume_ids = {r["id"] for rnd in rounds for r in rnd["resumes"]}
    jd_ids     = {j["id"] for rnd in rounds for j in rnd["jds"]}

    collected: dict[str, dict] = {}
    tried: set[str] = set()
    provider_fails: dict[str, int] = {}

    for attempt in range(1, MAX_ATTEMPTS_PER_STEP + 1):
        missing_ids = expected_ids - set(collected)
        if not missing_ids:
            break

        pair_prompt = build_group_pair_prompt(rounds, label_targets, current_dist, only_pair_ids=missing_ids)
        PROMPT_OUTPUT.write_text(pair_prompt, encoding="utf-8")

        raw, used_key = llm_client.call_llm(pair_prompt, exclude=tried)
        tried.add(used_key)
        TEMP_INPUT.write_text(raw, encoding="utf-8")
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]

        accepted, errors, warnings = _process_pair_lines(lines, rounds, missing_ids, resume_ids, jd_ids)
        collected.update(accepted)

        if warnings:
            print(f"  [auto] {len(warnings)} warning(s) (proceeding automatically):")
            for w in warnings[:5]:
                print(f"      ⚠ {w}")

        if accepted:
            # Forward progress — this provider clearly works on this prompt,
            # so don't let an earlier unrelated strike count against it.
            provider_fails[used_key.split(":", 1)[0]] = 0
            print(f"  [auto] attempt {attempt}/{MAX_ATTEMPTS_PER_STEP} via {used_key}: "
                  f"+{len(accepted)} pair(s) accepted ({len(collected)}/{N_PAIRS} total)")
        if len(accepted) < len(missing_ids):
            still_missing = len(missing_ids) - len(accepted)
            print(f"  [auto]   {still_missing} pair(s) still missing/invalid from this attempt:")
            for e in errors[:5]:
                print(f"      ✗ {e}")
            if not accepted:
                _record_provider_failure(used_key, provider_fails, tried)

    missing_ids = expected_ids - set(collected)
    if missing_ids:
        raise RuntimeError(
            f"Pair scoring: still missing {len(missing_ids)}/{N_PAIRS} pair(s) after "
            f"{MAX_ATTEMPTS_PER_STEP} attempts across different keys: {sorted(missing_ids)}"
        )

    print(f"  [auto] all {N_PAIRS} pairs scored.")
    return [collected[f"pair_{group_pair_start + i}"] for i in range(N_PAIRS)]


def _auto_run_one_group() -> None:
    session = load_session()

    if session.get("state") == "idle":
        _, _, group_pair_start = get_next_ids()
        rounds: list[dict] = []
    else:
        group_pair_start = session["group_pair_start"]
        rounds = session.get("rounds", [])

    if session.get("state") != "waiting_pairs":
        while len(rounds) < ROUNDS_PER_GROUP:
            round_index = len(rounds) + 1
            print(f"\n  [auto] generating round {round_index}/{ROUNDS_PER_GROUP}...")
            resumes_new, jds_new, round_meta = _auto_generate_round(round_index, group_pair_start)

            append_jsonl(RESUMES_PATH, resumes_new)
            append_jsonl(JDS_PATH,     jds_new)

            completed_round = dict(round_meta)
            completed_round["resumes"] = resumes_new
            completed_round["jds"]     = jds_new
            rounds = rounds + [completed_round]

            save_session({
                "state":            "waiting_cvjd",
                "group_pair_start": group_pair_start,
                "current_round":    round_meta,
                "rounds":           rounds,
            })
            print(f"  [auto] round {round_index}/{ROUNDS_PER_GROUP} saved.")

        save_session({
            "state":            "waiting_pairs",
            "group_pair_start": group_pair_start,
            "rounds":           rounds,
        })

    label_targets = compute_label_targets(N_PAIRS)
    current_dist  = dict(Counter(p.get("label") for p in load_jsonl(PAIRS_PATH) if p.get("label") in ALLOWED_LABELS))

    print(f"\n  [auto] all {ROUNDS_PER_GROUP} rounds ready — scoring {N_PAIRS} pairs...")
    pairs_new = _auto_generate_pairs(rounds, label_targets, current_dist, group_pair_start)

    append_jsonl(PAIRS_PATH, pairs_new)
    dist = Counter(p.get("label") for p in pairs_new)
    print(f"\n  [auto] ✓ saved {len(pairs_new)} pairs — " + "  |  ".join(f"{k}: {v}" for k, v in dist.items()))

    TEMP_INPUT.write_text("", encoding="utf-8")
    save_session({"state": "idle"})

    auto_commit_batch()
    show_stats()


def action_auto_run(max_groups: int | None = None) -> None:
    """Fully automated mode: build every prompt, call the configured API keys
    directly (automatic fallback across providers/keys on failure or
    exhaustion), validate, and save — no manual copy-paste required.

    Runs until `max_groups` groups are completed, or indefinitely (Ctrl+C to
    stop) if max_groups is None, or until every configured API key is
    exhausted — whichever happens first. Progress is saved after every round
    and every group, so an interrupted run can always be resumed with [1]/[2]
    (manual) or by choosing auto-run again.
    """
    if not _LLM_CLIENT_AVAILABLE:
        print("\n  [!] llm_client module unavailable (missing httpx / python-dotenv?). Cannot run auto mode.")
        return

    print("\n  Auto mode: calling configured API keys directly (no manual copy-paste).")
    if max_groups is None:
        print("  Press Ctrl+C to stop between groups.")

    groups_done = 0
    try:
        while max_groups is None or groups_done < max_groups:
            _auto_run_one_group()
            groups_done += 1
            suffix = f"/{max_groups}" if max_groups else ""
            print(f"\n  [auto] === group {groups_done}{suffix} complete ===")
    except llm_client.AllKeysExhaustedError as e:
        print(f"\n  [!] {e}")
        print("  [!] Stopping — no more usable API keys. Progress so far is saved.")
    except KeyboardInterrupt:
        print("\n\n  Stopped by user (Ctrl+C). Progress so far is saved — resume anytime.")
    except RuntimeError as e:
        print(f"\n  [!] {e}")
        print("  [!] Stopping — inspect scripts/temp_input.txt for the last LLM response.")

# ── main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║      Synthetic CV-JD Data Generator  (v3)        ║")
    print(f"║  {N_RESUMES}x{N_JDS} rounds × {ROUNDS_PER_GROUP} = {N_PAIRS} pairs scored per group        ║")
    print("╚══════════════════════════════════════════════════╝")
    show_stats()

    # --auto [N]: skip the interactive menu entirely and run N groups (or
    # indefinitely if N is omitted) via the configured API keys.
    if len(sys.argv) > 1 and sys.argv[1] in ("--auto", "-a"):
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        action_auto_run(n)
        return

    while True:
        session = load_session()
        state   = session.get("state", "idle")

        print("\nMenu:")
        print("  [1]  Generate CV+JD prompt  (start new group / next round)")
        print("  [2]  Save result            (paste LLM output into scripts/temp_input.txt first)")
        print("  [3]  Auto-run via API       (no manual copy-paste, uses keys from .env)")
        print("  [0]  Exit")

        if state != "idle":
            round_num = session.get("current_round", {}).get("index", len(session.get("rounds", [])))
            print(f"\n  Current state : {state}")
            print(f"  Round         : {round_num}/{ROUNDS_PER_GROUP}" if state == "waiting_cvjd" else f"  Round         : all {ROUNDS_PER_GROUP} done, waiting for pair scores")

        choice = input("\n  Choice: ").strip()

        if choice == "1":
            action_generate_prompt()
        elif choice == "2":
            action_save()
        elif choice == "3":
            raw_n = input("      How many groups? (blank = run until stopped or keys exhausted): ").strip()
            action_auto_run(int(raw_n) if raw_n else None)
        elif choice == "0":
            print("\n  Bye.\n")
            sys.exit(0)
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
