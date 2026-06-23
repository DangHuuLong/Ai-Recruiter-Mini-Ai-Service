"""
Synthetic data generator for CV-JD pairs.

Workflow per batch (25 pairs from 5 resumes × 5 JDs):
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

RESUMES_PATH = PROJECT_ROOT / "datasets" / "raw" / "resumes.jsonl"
JDS_PATH     = PROJECT_ROOT / "datasets" / "raw" / "job_descriptions.jsonl"
PAIRS_PATH   = PROJECT_ROOT / "datasets" / "processed" / "cv_jd_pairs.jsonl"
TEMP_INPUT     = SCRIPTS_DIR / "temp_input.txt"
PROMPT_OUTPUT  = SCRIPTS_DIR / "prompt_output.txt"
SESSION_FILE   = SCRIPTS_DIR / ".synthetic_session.json"

# ── constants ─────────────────────────────────────────────────────────────────

BOUNDARY_ZONES   = [(35, 45), (55, 65), (70, 80), (85, 95)]
FORBIDDEN_LABELS = {"poor_match"}
ALLOWED_LABELS   = {"weak_match", "moderate_match", "strong_match", "excellent_match"}

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
    print("\n" + "=" * 56)
    print(f"  Resumes: {len(resumes)}   JDs: {len(jds)}   Pairs: {total}")
    print("  Label distribution:")
    for label in labels_order:
        count = dist.get(label, 0)
        pct   = count / total * 100 if total else 0
        bar   = "█" * int(pct / 2.5)
        print(f"    {label:<20} {count:>5}  ({pct:4.1f}%)  {bar}")

    target = max((dist.get(label, 0) for label in ALLOWED_LABELS), default=0)
    mod_need = max(0, target - dist.get("moderate_match", 0))
    str_need = max(0, target - dist.get("strong_match", 0))
    exc_need = max(0, target - dist.get("excellent_match", 0))
    total_need = mod_need + str_need + exc_need
    print(f"\n  Still needed to balance (target = {target} each):")
    print(f"    moderate_match  +{mod_need:>4}  (~{mod_need  // 25 + (1 if mod_need  % 25 else 0):>2} groups)")
    print(f"    strong_match    +{str_need:>4}  (~{str_need  // 25 + (1 if str_need  % 25 else 0):>2} groups)")
    print(f"    excellent_match +{exc_need:>4}  (~{exc_need  // 25 + (1 if exc_need  % 25 else 0):>2} groups)")
    print(f"    Total new pairs needed: ~{total_need}")
    print("=" * 56)


def compute_label_targets(n_pairs: int = 25) -> dict[str, int]:
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

    # Inverse weight: label with fewer pairs gets higher weight
    max_count = max(dist.get(label, 0) for label in labels)
    # +1 so labels at max_count still get a small weight
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

def build_cvjd_prompt(domain: str, resume_start: int, jd_start: int) -> str:
    r_ids = [f"resume_{resume_start + i}" for i in range(5)]
    j_ids = [f"jd_{jd_start + i}"        for i in range(5)]
    levels = ["intern", "junior", "middle", "senior", "senior"]

    return f"""You are a dataset generator for an AI recruitment system.
Generate exactly 5 resumes and 5 job descriptions in the domain: {domain}

═══ CRITICAL CONSTRAINT ═══
These 10 documents will be cross-paired into 25 CV-JD pairs.
Target score range: 40–100 (weak / moderate / strong / excellent match).
ZERO pairs may score below 40 (no poor_match allowed).

Place ALL resumes and JDs in the SAME domain ({domain}).
Vary seniority level so cross-pairings naturally spread across the score range:
extreme mismatches (intern ↔ senior lead) may score 40–59 (weak_match),
while same-level pairings should score 75–100 (strong / excellent).

Seniority ladder to use (one level per document):
  Level 1 → intern   (0.0 – 0.5 yrs)
  Level 2 → junior   (1.0 – 2.0 yrs)
  Level 3 → middle   (3.0 – 5.0 yrs)
  Level 4 → senior   (5.0 – 7.0 yrs)
  Level 5 → senior   (7.0 + yrs, lead/principal role)

═══ OUTPUT FORMAT ═══
Output exactly 10 lines.
Each line is a single JSON object (no array brackets, no commas between lines).
Lines 1–5: resumes (IDs: {r_ids[0]} → {r_ids[4]}, levels: {levels})
Lines 6–10: job descriptions (IDs: {j_ids[0]} → {j_ids[4]}, levels: {levels})
No markdown, no explanation, no extra text — only 10 JSON lines.

═══ RESUME SCHEMA (lines 1–5) ═══
{{"id":"<resume_id>","source":"synthetic","candidate_level":"<intern|junior|middle|senior>","raw_text":"<150–250 word prose — no names, emails, phone numbers, LinkedIn or GitHub URLs>","summary":"<one sentence>","skills":["<TitleCased>",...],"normalized_skills":["<snake_case>",...],"experience_years":<float>,"education":{{"degree":"<Bachelor|Master|PhD>","field_of_study":"<field>","status":"<student|final_year|graduate>"}},"projects":[{{"name":"<name>","description":"<1–2 sentences>","technologies":["<Tech>",...]}}],"certifications":[],"languages":[{{"name":"English","proficiency":"intermediate"}}],"anonymized":true,"language":"en"}}

═══ JD SCHEMA (lines 6–10) ═══
{{"id":"<jd_id>","source":"synthetic","title":"<Job Title>","level":"<intern|junior|middle|senior>","employment_type":"<internship|full_time|contract>","location":"<City, Country or Remote>","remote_allowed":<true|false>,"posted_date":null,"domain":"<slug>","raw_text":"<100–180 word prose — no company name>","responsibilities":["<item>",...],"requirements":["<item>",...],"nice_to_have":["<item>",...],"required_skills":["<TitleCased>",...],"preferred_skills":["<TitleCased>",...],"min_experience_years":<int>,"education_requirement":"<string>","domain_keywords":["<snake_case>",...],"language":"en"}}"""


def build_pair_prompt(
    resumes: list[dict],
    jds: list[dict],
    pair_start: int,
    label_targets: dict[str, int] | None = None,
) -> str:
    def resume_summary(r: dict) -> str:
        return (
            f"  id: {r['id']} | level: {r.get('candidate_level')} | "
            f"exp: {r.get('experience_years')} yrs\n"
            f"  skills: {', '.join(r.get('skills', []))}\n"
            f"  background: {r.get('raw_text', '')[:200]}"
        )

    def jd_summary(j: dict) -> str:
        return (
            f"  id: {j['id']} | title: {j.get('title')} | level: {j.get('level')} | "
            f"min_exp: {j.get('min_experience_years')} yrs\n"
            f"  required_skills: {', '.join(j.get('required_skills', []))}\n"
            f"  description: {j.get('raw_text', '')[:180]}"
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
        f"  pair_{pair_start + i:<4}  {rid} × {jid}"
        for i, (_, rid, jid) in enumerate(pair_ids)
    )

    if label_targets:
        label_order = ["strong_match", "excellent_match", "moderate_match", "weak_match"]
        dist_lines = "\n".join(
            f"  {label}: ~{label_targets.get(label, 0)} pairs"
            for label in label_order
            if label in label_targets
        )
        distribution_section = f"""
═══ TARGET LABEL DISTRIBUTION ═══
The dataset is currently imbalanced. Bias your scoring to hit these counts.
strong_match is the PRIMARY target — give it the highest allocation.
{dist_lines}
Adjust criterion scores within valid ranges to reach these targets while
keeping every score honest and consistent with the documents.
"""
    else:
        distribution_section = ""

    return f"""You are a dataset labeler for an AI recruitment system.
Score 25 CV-JD pairs using rubric v0.2.

═══ RESUMES ═══
{resume_block}

═══ JOB DESCRIPTIONS ═══
{jd_block}

═══ SCORING RUBRIC v0.2 ═══
Score each criterion 0–100, then compute:
  overall_score = round(SKILLS_MATCH×0.35 + EXPERIENCE_RELEVANCE×0.30 +
                        PROJECT_RELEVANCE×0.15 + EDUCATION_CERTIFICATION×0.10 +
                        KEYWORD_DOMAIN_ALIGNMENT×0.10)

Label mapping:
  90–100 → excellent_match
  75–89  → strong_match       ← PRIMARY TARGET — maximise this label
  60–74  → moderate_match
  40–59  → weak_match
  0–39   → poor_match         ← FORBIDDEN — do not generate

Scores below 40 are forbidden. Floor is 40.
Prioritise strong_match (75–89): aim for at least half the pairs in this range.

Avoid boundary zones — do NOT use scores in these ranges:
  35–45, 55–65, 70–80, 85–95
Use clear mid-range scores instead: e.g. 48, 65, 77, 83, 92.
{distribution_section}
═══ PAIRING ORDER (25 pairs) ═══
{pair_order}

═══ OUTPUT FORMAT ═══
Output exactly 25 lines.
Each line is a single JSON object (no array brackets, no commas between lines).
No markdown, no explanation, no extra text — only 25 JSON lines.

JSON schema per pair:
{{"id":"<pair_id>","resume_id":"<resume_id>","job_description_id":"<jd_id>","split":null,"label":"<weak_match|moderate_match|strong_match|excellent_match>","overall_score":<int 40–100>,"criterion_scores":{{"SKILLS_MATCH":<int>,"EXPERIENCE_RELEVANCE":<int>,"PROJECT_RELEVANCE":<int>,"EDUCATION_CERTIFICATION":<int>,"KEYWORD_DOMAIN_ALIGNMENT":<int>}},"matched_skills":["<skill>",...],"missing_required_skills":["<skill>",...],"matched_preferred_skills":["<skill>",...],"label_notes":"<1–2 sentences: strongest match point and key gap>","labeled_by":"llm_synthetic","label_version":"rubric_v0.2"}}"""

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

    if label in FORBIDDEN_LABELS:
        errs.append(f"pair {pid}: label '{label}' is forbidden (poor_match not allowed in synthetic data)")
    elif label not in ALLOWED_LABELS:
        errs.append(f"pair {pid}: unknown label '{label}'")

    if score is None:
        errs.append(f"pair {pid}: missing overall_score")
    else:
        score = int(score)
        if score < 40:
            errs.append(f"pair {pid}: overall_score {score} is below 40 — forbidden")
        if in_boundary_zone(score):
            warns.append(f"pair {pid}: score {score} is in a boundary zone {BOUNDARY_ZONES}")
        if label in SCORE_RANGES:
            lo, hi = SCORE_RANGES[label]
            if not (lo <= score <= hi):
                errs.append(f"pair {pid}: score {score} does not match label '{label}' (expected {lo}–{hi})")

    cs = p.get("criterion_scores", {})
    for key in CRITERIA_KEYS:
        if key not in cs:
            errs.append(f"pair {pid}: criterion_scores missing '{key}'")
        elif not isinstance(cs[key], int) or not (0 <= cs[key] <= 100):
            errs.append(f"pair {pid}: criterion_scores.{key} must be int 0–100")

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
    expected_resume_ids = {f"resume_{resume_start + i}" for i in range(5)}
    expected_jd_ids     = {f"jd_{jd_start + i}"        for i in range(5)}

    actual_resume_ids = {r.get("id", f"<missing id on resume #{i}>") for i, r in enumerate(resumes)}
    actual_jd_ids     = {j.get("id", f"<missing id on jd #{i}>")     for i, j in enumerate(jds)}

    for rid in sorted(expected_resume_ids - actual_resume_ids):
        errs.append(f"[ID] resume '{rid}' expected but missing from LLM output")
    for rid in sorted(actual_resume_ids - expected_resume_ids):
        errs.append(
            f"[ID] resume '{rid}' is unexpected "
            f"(expected resume_{resume_start}–resume_{resume_start + 4})"
        )
    for jid in sorted(expected_jd_ids - actual_jd_ids):
        errs.append(f"[ID] JD '{jid}' expected but missing from LLM output")
    for jid in sorted(actual_jd_ids - expected_jd_ids):
        errs.append(
            f"[ID] JD '{jid}' is unexpected "
            f"(expected jd_{jd_start}–jd_{jd_start + 4})"
        )
    return errs


def validate_pair_ids(pairs: list[dict], pair_start: int) -> list[str]:
    errs: list[str] = []
    expected_pair_ids = {f"pair_{pair_start + i}" for i in range(25)}

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
            f"(expected pair_{pair_start}–pair_{pair_start + 24})"
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

    prompt = build_cvjd_prompt(domain, resume_start, jd_start)

    save_session({
        "state":        "waiting_cvjd",
        "domain":       domain,
        "resume_start": resume_start,
        "jd_start":     jd_start,
        "pair_start":   pair_start,
    })

    # Ensure temp_input.txt exists and is empty
    TEMP_INPUT.write_text("", encoding="utf-8")

    print(f"\n  Domain  : {domain}")
    print(f"  IDs     : resume_{resume_start}–{resume_start+4}  |  jd_{jd_start}–{jd_start+4}")
    PROMPT_OUTPUT.write_text(prompt, encoding="utf-8")
    print(f"\n  ✓ Prompt written to:  scripts/prompt_output.txt")
    print(f"  → Open the file, copy all content, paste into your LLM.")
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
        if len(lines) != 10:
            print(f"\n  [!] Expected 10 lines (5 resumes + 5 JDs), got {len(lines)}.")
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

        if len(resumes_new) != 5 or len(jds_new) != 5:
            print(f"\n  [!] Expected 5 resumes + 5 JDs, got {len(resumes_new)} + {len(jds_new)}.")
            return

        # Validate IDs first
        id_errors = validate_cvjd_ids(
            resumes_new, jds_new,
            session["resume_start"], session["jd_start"],
        )
        if id_errors:
            print(f"\n  [!] {len(id_errors)} ID error(s) — NOT saved:")
            for e in id_errors:
                print(f"      ✗ {e}", file=sys.stderr)
                print(f"      ✗ {e}")
            print("\n      Fix the IDs in temp_input.txt and press [2] again.")
            return

        # Validate fields
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

        # Save
        append_jsonl(RESUMES_PATH, resumes_new)
        append_jsonl(JDS_PATH,     jds_new)
        print(f"\n  ✓ Saved {len(resumes_new)} resumes → {RESUMES_PATH.name}")
        print(f"  ✓ Saved {len(jds_new)} JDs      → {JDS_PATH.name}")

        # Update session and auto-generate pair prompt
        session["state"]   = "waiting_pairs"
        session["resumes"] = resumes_new
        session["jds"]     = jds_new
        save_session(session)

        TEMP_INPUT.write_text("", encoding="utf-8")

        label_targets = compute_label_targets(25)
        pair_prompt = build_pair_prompt(resumes_new, jds_new, session["pair_start"], label_targets)
        print(f"  Label targets this batch: " + "  ".join(f"{k}: {v}" for k, v in sorted(label_targets.items())))
        PROMPT_OUTPUT.write_text(pair_prompt, encoding="utf-8")
        print(f"\n  ✓ Pair prompt written to:  scripts/prompt_output.txt")
        print(f"  → Open the file, copy all content, paste into your LLM.")
        print(f"  → Paste LLM output into:  scripts/temp_input.txt")
        print(f"  → Then press [2] to save pairs.")

    # ── save pairs ────────────────────────────────────────────────────────────
    elif state == "waiting_pairs":
        if len(lines) != 25:
            print(f"\n  [!] Expected 25 lines (25 pairs), got {len(lines)}.")
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

        # Validate pair IDs
        pair_id_errors = validate_pair_ids(pairs_new, session["pair_start"])
        if pair_id_errors:
            print(f"\n  [!] {len(pair_id_errors)} pair ID error(s) — NOT saved:")
            for e in pair_id_errors:
                print(f"      ✗ {e}", file=sys.stderr)
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

        # Save pairs
        append_jsonl(PAIRS_PATH, pairs_new)
        print(f"\n  ✓ Saved {len(pairs_new)} pairs → {PAIRS_PATH.name}")

        # Summarise this batch
        dist = Counter(p.get("label") for p in pairs_new)
        print(f"  Batch label breakdown: ", end="")
        print("  |  ".join(f"{k}: {v}" for k, v in dist.items()))

        # Reset
        TEMP_INPUT.write_text("", encoding="utf-8")
        save_session({"state": "idle"})

        show_stats()

# ── main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║      Synthetic CV-JD Data Generator              ║")
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
