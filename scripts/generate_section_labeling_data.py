"""
Synthetic section-labeling data generator for the resume section-splitter
distillation project (see C:\\Users\\HELLO\\.claude\\plans\\tingly-foraging-quill.md).

This is the OPPOSITE of scripts/generate_synthetic_data.py's CV prompt: that
script forbids bullet points and section headers (prose-only, for CV-JD
scoring). This script REQUIRES them — headers, bullets, varied layout,
occasional no-header resumes — because the student model being trained here
(training/fine_tune_section_classifier.py) needs to learn header/bullet/
layout boundary cues that the existing regex splitter's fixed alias list
cannot generalize to.

Each LLM call asks for a small batch of resumes, each pre-assigned a
header_style / language / hard_case / domain (rotated for diversity), and
gets back one JSON object per resume: the resume's lines, each labeled with
the resume section it belongs to (see app.parsers.section_splitter.SECTION_ORDER
for the label vocabulary — imported directly so it can never drift from the
splitter's contract).

Retries only re-request whatever resumes are still missing/invalid from a
batch (same accumulate-and-retry-only-missing pattern as
scripts/generate_synthetic_data.py's _auto_generate_pairs), and progress is
resumable via .section_labeling_session.json.

Run:
  python scripts/generate_section_labeling_data.py --auto [N_BATCHES]
  python scripts/generate_section_labeling_data.py --auto --target 500
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
sys.path.insert(0, str(PROJECT_ROOT))  # so `import app...` works regardless of cwd

from app.parsers.section_splitter import SECTION_ORDER  # noqa: E402

OUTPUT_DIR       = PROJECT_ROOT / "datasets" / "section_splitting" / "v0.1"
OUTPUT_PATH      = OUTPUT_DIR / "labeled_lines.jsonl"
SESSION_FILE     = SCRIPTS_DIR / ".section_labeling_session.json"
STYLE_USAGE_FILE = SCRIPTS_DIR / ".section_style_usage.json"
TEMP_INPUT       = SCRIPTS_DIR / "section_labeling_temp_input.txt"
PROMPT_OUTPUT    = SCRIPTS_DIR / "section_labeling_prompt_output.txt"

# ── constants ─────────────────────────────────────────────────────────────────

ALLOWED_LABELS = set(SECTION_ORDER) | {"other"}

BATCH_SIZE = 3  # resumes per LLM call — kept small since each resume's output
                # is a full line-by-line JSON array, more verbose per-item
                # than the prose CVs in generate_synthetic_data.py.

MAX_ATTEMPTS_PER_STEP = 12
PROVIDER_STRIKES_BEFORE_SKIP = 2

MIN_LINES_PER_RESUME  = 15
MIN_DISTINCT_LABELS   = 4

HEADER_STYLES = [
    "ALL CAPS headers on their own line (e.g. \"EXPERIENCE\")",
    "Title Case headers on their own line (e.g. \"Work Experience\")",
    "Markdown-bold headers (e.g. \"**Skills**\")",
    "inline label-colon style with no standalone header line (e.g. \"Skills: Python, SQL, Docker\")",
    "minimal-to-no explicit headers — sections must be distinguishable mainly from content/context, not header text",
]

LANGUAGES = ["en", "vi", "mixed"]

HARD_CASES = [
    "none - straightforward single-column layout",
    "stacked project dates - list 2 project titles/descriptions first, then bunch both of their date ranges together at the very end of the Projects section (simulates a PDF-extraction column artifact)",
    "wrapped skill line - the Skills section's tools list wraps across 2-3 separate lines with no comma or punctuation cue at the line break that it's a continuation",
    "two-column artifact - interleave two projects' bullet lines as if a 2-column PDF layout collapsed into one column (alternate: project A detail, project B detail, project A detail, ...)",
]

# Independent, smaller domain list — intentionally NOT shared with
# generate_synthetic_data.py's DOMAINS/.domain_usage.json (that rotation is
# scoped to the CV-JD scoring dataset; this is a different dataset/purpose).
DOMAINS = [
    "Software Engineering", "Data Analysis", "Digital Marketing", "Finance and Accounting",
    "Human Resources", "Sales", "Customer Support", "Project Management", "UI/UX Design",
    "Mobile Development", "DevOps", "Product Management", "Business Analysis", "Content Writing",
    "Supply Chain", "Legal", "Education and Training", "Healthcare Administration",
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE   = re.compile(r"https?://(?:www\.)?(linkedin|github)\.com/[^\s]+", re.I)

LABEL_LEGEND = """  summary          intro/profile/objective/about-me paragraph
  skills           technical and soft skills, tools, technologies
  experience       work history entries (job title, company type, duration, duties)
  projects         personal/side/work projects (name, description, tech, outcome)
  education        degrees, schools, graduation year
  certifications   certificates, licenses
  achievements     awards, honors, recognitions
  languages        spoken language proficiency (NOT programming languages — those go under skills)
  other            use ONLY for lines with no clear section membership (e.g. a
                   decorative separator like "----"). Do NOT use "other" for
                   content that clearly belongs to a section just because
                   you're unsure of exact wording — pick the best-fit section."""

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

# ── style/domain rotation (least-used, persisted) ─────────────────────────────

def _load_usage() -> dict:
    if STYLE_USAGE_FILE.exists():
        try:
            return json.loads(STYLE_USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_usage(usage: dict) -> None:
    STYLE_USAGE_FILE.write_text(json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8")


def pick_rotating(usage: dict, axis: str, options: list[str]) -> str:
    """Pick the least-used option for `axis` so far (random tie-break),
    mirroring generate_synthetic_data.py's pick_domain() rotation — plain
    random.choice would let a handful of styles dominate across many batches."""
    counts = usage.setdefault(axis, {})
    min_count = min((counts.get(o, 0) for o in options), default=0)
    least_used = [o for o in options if counts.get(o, 0) == min_count]
    chosen = random.choice(least_used)
    counts[chosen] = counts.get(chosen, 0) + 1
    return chosen

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
    """Stage and commit this batch's files, scoped to exactly what this script
    writes (not `git add .`) — mirrors generate_synthetic_data.py's
    auto_commit_batch() so it never sweeps up unrelated in-progress edits."""
    files = [OUTPUT_PATH, TEMP_INPUT, PROMPT_OUTPUT, SESSION_FILE, STYLE_USAGE_FILE]
    try:
        subprocess.run(
            ["git", "add", "--", *[str(f) for f in files]],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", "feat: add section-labeling training data batch"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("\n  ✓ Committed to git")
        else:
            print(f"\n  [!] git commit skipped: {(result.stdout or result.stderr).strip()}")
    except subprocess.CalledProcessError as e:
        print(f"\n  [!] git add failed: {(e.stderr or str(e)).strip()}")
    except FileNotFoundError:
        print("\n  [!] git not found on PATH — skipped auto-commit")

# ── stats ──────────────────────────────────────────────────────────────────────

def show_stats() -> None:
    records = load_jsonl(OUTPUT_PATH)
    total_lines = sum(len(r.get("lines", [])) for r in records)
    label_counts: Counter = Counter()
    for r in records:
        for item in r.get("lines", []):
            if isinstance(item, dict):
                label_counts[item.get("label")] += 1

    print("\n" + "=" * 60)
    print(f"  Labeled resumes: {len(records)}   Total labeled lines: {total_lines}")
    if total_lines:
        print("  Label distribution (lines):")
        for label in sorted(ALLOWED_LABELS):
            count = label_counts.get(label, 0)
            pct = count / total_lines * 100
            print(f"    {label:<16} {count:>6}  ({pct:4.1f}%)")
    print("=" * 60)

# ── prompt builder ─────────────────────────────────────────────────────────────

def _next_start_id() -> int:
    return get_max_numeric_id(load_jsonl(OUTPUT_PATH), "resline_") + 1


def _build_batch_specs(start_id: int, batch_size: int = BATCH_SIZE) -> list[dict]:
    usage = _load_usage()
    specs = []
    for i in range(batch_size):
        specs.append({
            "id":           f"resline_{start_id + i:04d}",
            "header_style": pick_rotating(usage, "header_style", HEADER_STYLES),
            "language":     pick_rotating(usage, "language", LANGUAGES),
            "hard_case":    pick_rotating(usage, "hard_case", HARD_CASES),
            "domain":       pick_rotating(usage, "domain", DOMAINS),
        })
    _save_usage(usage)
    return specs


def build_labeling_prompt(specs: list[dict], only_ids: set[str] | None = None) -> str:
    if only_ids is not None:
        specs = [s for s in specs if s["id"] in only_ids]

    assignment_lines = "\n\n".join(
        f"  {s['id']}:\n"
        f"    domain       : {s['domain']}\n"
        f"    header_style : {s['header_style']}\n"
        f"    language     : {s['language']}\n"
        f"    hard_case    : {s['hard_case']}"
        for s in specs
    )

    partial_note = ""
    if only_ids is not None:
        partial_note = """
════════════════════════════════════
PARTIAL RE-REQUEST
════════════════════════════════════
A previous response was incomplete. Output ONLY the resumes listed below —
do not re-output or repeat any other resume.
"""

    return f"""You are generating training data to teach a small model how real
resumes are laid out into sections. This is DIFFERENT from other resume
generation you may have done before: those wanted clean prose with NO
headers or bullets. This task wants the OPPOSITE — realistic resumes with
EXPLICIT section headers and bullet points (except where a resume's
assignment below calls for a minimal/no-header style), because the model
being trained needs real header/bullet/layout cues to learn from.
{partial_note}
════════════════════════════════════
RESUMES TO GENERATE ({len(specs)})
════════════════════════════════════
{assignment_lines}

For each resume: write realistic content for a mid-career professional in
the given domain (any seniority — vary it). Do NOT include a candidate name,
email, phone number, LinkedIn URL, or GitHub URL anywhere in the text.

════════════════════════════════════
LAYOUT REQUIREMENTS PER header_style
════════════════════════════════════
- "ALL CAPS headers": each section starts with its name in ALL CAPS on its
  own line (e.g. "EXPERIENCE"), followed by bullet-point or short-line content.
- "Title Case headers": same but Title Case (e.g. "Work Experience").
- "Markdown-bold headers": headers wrapped in ** (e.g. "**Skills**").
- "inline label-colon style": no standalone header line at all — each
  section's content starts right after "Label: " on the same line
  (e.g. "Skills: Python, Django, PostgreSQL, Docker").
- "minimal-to-no explicit headers": write the resume as a sequence of
  clearly distinguishable content blocks (a skills line still reads like a
  skills line, an experience entry still reads like a job entry) but WITHOUT
  a labeled header line introducing most sections.
Apply the hard_case for each resume literally where it's not "none" — these
are deliberately awkward/realistic layout quirks the model must learn to
handle, do not smooth them over.

════════════════════════════════════
LINE-LEVEL LABELING — every line gets exactly one label
════════════════════════════════════
{LABEL_LEGEND}

A header line itself (e.g. the line that just says "EXPERIENCE") gets the
SAME label as the section it introduces — do not invent a separate "header"
label. Bullet lines keep their bullet marker (e.g. "- Built a...") in `text`.
Blank/empty lines should simply be omitted from the `lines` array (don't
emit empty-string entries).

Each resume needs at least {MIN_LINES_PER_RESUME} lines total and at least
{MIN_DISTINCT_LABELS} distinct section labels represented (skills alone
isn't enough — include experience, education, etc. as make sense for a
real resume).

════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════
Output exactly {len(specs)} lines. Each line is a single JSON object (no
array brackets, no commas between lines, no markdown fences, no explanation).

Schema per resume:
{{"id":"<resline_id>","lines":[{{"text":"<line text, bullet marker included if present>","label":"<summary|skills|experience|projects|education|certifications|achievements|languages|other>"}}, ...]}}
"""

# ── validation ─────────────────────────────────────────────────────────────────

def validate_labeled_resume(obj: dict) -> list[str]:
    rid = obj.get("id", "?")
    errs: list[str] = []

    for field in ("id", "lines"):
        if field not in obj:
            errs.append(f"resume {rid}: missing field '{field}'")

    lines = obj.get("lines")
    if not isinstance(lines, list) or not lines:
        errs.append(f"resume {rid}: 'lines' must be a non-empty list")
        return errs

    if len(lines) < MIN_LINES_PER_RESUME:
        errs.append(f"resume {rid}: only {len(lines)} lines, minimum {MIN_LINES_PER_RESUME}")

    labels_seen: set[str] = set()
    full_text_parts: list[str] = []
    for i, item in enumerate(lines, 1):
        if not isinstance(item, dict) or "text" not in item or "label" not in item:
            errs.append(f"resume {rid} line {i}: must be an object with 'text' and 'label'")
            continue
        if not isinstance(item["text"], str) or not item["text"].strip():
            errs.append(f"resume {rid} line {i}: 'text' must be a non-empty string")
        if item.get("label") not in ALLOWED_LABELS:
            errs.append(f"resume {rid} line {i}: invalid label '{item.get('label')}'")
        else:
            labels_seen.add(item["label"])
        full_text_parts.append(str(item.get("text", "")))

    if len(labels_seen) < MIN_DISTINCT_LABELS:
        errs.append(
            f"resume {rid}: only {len(labels_seen)} distinct section label(s) used "
            f"({sorted(labels_seen)}), minimum {MIN_DISTINCT_LABELS}"
        )

    joined = "\n".join(full_text_parts)
    if EMAIL_RE.search(joined):
        errs.append(f"resume {rid}: contains an email-like pattern")
    if URL_RE.search(joined):
        errs.append(f"resume {rid}: contains a LinkedIn/GitHub URL pattern")

    return errs


def _process_labeling_lines(
    lines: list[str], expected_ids: set[str],
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Parse + validate labeling-response lines against `expected_ids`. Mirrors
    generate_synthetic_data.py's _process_pair_lines: returns (accepted,
    errors, warnings) where `accepted` only contains ids that both parsed and
    passed validation in THIS response — missing/failing ids are reported in
    `errors` for the caller to retry, not treated as fatal for the whole batch."""
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
        rid = obj.get("id")
        if rid not in expected_ids:
            warnings.append(f"[ID] resume '{rid}' is unexpected/extra — dropped")
            continue
        if rid in by_id:
            warnings.append(f"[ID] resume '{rid}' is duplicated — keeping first occurrence")
            continue
        by_id[rid] = obj

    errors: list[str] = list(parse_errors)
    accepted: dict[str, dict] = {}
    for rid, obj in by_id.items():
        errs = validate_labeled_resume(obj)
        if errs:
            errors.extend(errs)
            continue
        accepted[rid] = obj

    for rid in sorted(expected_ids - set(by_id)):
        errors.append(f"[ID] resume '{rid}' expected but missing from LLM output")

    return accepted, errors, warnings

# ── fully-automated mode (calls .env API keys directly) ───────────────────────

try:
    import llm_client
    _LLM_CLIENT_AVAILABLE = True
except ImportError:
    _LLM_CLIENT_AVAILABLE = False


def _record_provider_failure(used_key: str, provider_fails: dict[str, int], tried: set[str]) -> None:
    """Same policy as generate_synthetic_data.py: don't give up on a whole
    provider after one bad sample (temperature variance), only after
    PROVIDER_STRIKES_BEFORE_SKIP consecutive failures from that provider."""
    provider = used_key.split(":", 1)[0]
    provider_fails[provider] = provider_fails.get(provider, 0) + 1
    if provider_fails[provider] >= PROVIDER_STRIKES_BEFORE_SKIP:
        tried |= llm_client.provider_tokens(used_key)


def _auto_generate_batch(specs: list[dict]) -> list[dict]:
    """Generate + label one batch, accumulating across retries: each attempt
    only asks for whatever resumes are still missing, and validated resumes
    from a partial response are kept rather than discarded (same pattern as
    generate_synthetic_data.py's _auto_generate_pairs)."""
    expected_ids = {s["id"] for s in specs}
    collected: dict[str, dict] = {}
    tried: set[str] = set()
    provider_fails: dict[str, int] = {}

    for attempt in range(1, MAX_ATTEMPTS_PER_STEP + 1):
        missing_ids = expected_ids - set(collected)
        if not missing_ids:
            break

        prompt = build_labeling_prompt(specs, only_ids=missing_ids)
        PROMPT_OUTPUT.write_text(prompt, encoding="utf-8")

        raw, used_key = llm_client.call_llm(prompt, exclude=tried)
        tried.add(used_key)
        TEMP_INPUT.write_text(raw, encoding="utf-8")
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]

        accepted, errors, warnings = _process_labeling_lines(lines, missing_ids)
        collected.update(accepted)

        if warnings:
            print(f"  [auto] {len(warnings)} warning(s):")
            for w in warnings[:5]:
                print(f"      ⚠ {w}")

        if accepted:
            provider_fails[used_key.split(":", 1)[0]] = 0
            print(f"  [auto] attempt {attempt}/{MAX_ATTEMPTS_PER_STEP} via {used_key}: "
                  f"+{len(accepted)} resume(s) accepted ({len(collected)}/{len(specs)} total)")
        if len(accepted) < len(missing_ids):
            still_missing = len(missing_ids) - len(accepted)
            print(f"  [auto]   {still_missing} resume(s) still missing/invalid:")
            for e in errors[:5]:
                print(f"      ✗ {e}")
            if not accepted:
                _record_provider_failure(used_key, provider_fails, tried)

    missing_ids = expected_ids - set(collected)
    if missing_ids:
        raise RuntimeError(
            f"Batch: still missing {len(missing_ids)}/{len(specs)} resume(s) after "
            f"{MAX_ATTEMPTS_PER_STEP} attempts across different keys: {sorted(missing_ids)}"
        )

    return [collected[s["id"]] for s in specs]


def _auto_run_one_batch() -> None:
    start_id = _next_start_id()
    specs = _build_batch_specs(start_id)
    print(f"\n  [auto] generating batch: {specs[0]['id']}..{specs[-1]['id']}")
    for s in specs:
        print(f"    {s['id']}: {s['domain']} | {s['header_style'][:40]}... | {s['language']} | {s['hard_case'][:30]}...")

    resumes_new = _auto_generate_batch(specs)
    append_jsonl(OUTPUT_PATH, resumes_new)
    total_lines = sum(len(r["lines"]) for r in resumes_new)
    print(f"  [auto] ✓ saved {len(resumes_new)} resume(s), {total_lines} labeled lines → {OUTPUT_PATH.name}")

    auto_commit_batch()


def action_auto_run(max_batches: int | None, target_total: int | None) -> None:
    """Runs until `max_batches` batches are completed, or `target_total`
    resumes exist in OUTPUT_PATH, or indefinitely (Ctrl+C to stop) if both are
    None, or until every configured API key is exhausted — whichever first.
    Progress is saved after every batch, so an interrupted run resumes
    exactly where it left off next time."""
    if not _LLM_CLIENT_AVAILABLE:
        print("\n  [!] llm_client module unavailable (missing httpx / python-dotenv?). Cannot run auto mode.")
        return

    print("\n  Auto mode: calling configured API keys directly (no manual copy-paste).")
    if max_batches is None and target_total is None:
        print("  Press Ctrl+C to stop between batches.")

    batches_done = 0
    try:
        while True:
            if max_batches is not None and batches_done >= max_batches:
                break
            if target_total is not None and len(load_jsonl(OUTPUT_PATH)) >= target_total:
                print(f"\n  [auto] target of {target_total} resumes reached.")
                break

            _auto_run_one_batch()
            batches_done += 1
            show_stats()
    except llm_client.AllKeysExhaustedError as e:
        print(f"\n  [!] {e}")
        print("  [!] Stopping — no more usable API keys. Progress so far is saved.")
    except KeyboardInterrupt:
        print("\n\n  Stopped by user (Ctrl+C). Progress so far is saved — resume anytime.")
    except RuntimeError as e:
        print(f"\n  [!] {e}")
        print(f"  [!] Stopping — inspect {TEMP_INPUT.name} for the last LLM response.")

# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Section-Labeling Data Generator (distillation)     ║")
    print(f"║   batch size: {BATCH_SIZE} resumes/call                          ║")
    print("╚══════════════════════════════════════════════════════╝")
    show_stats()

    if len(sys.argv) > 1 and sys.argv[1] in ("--auto", "-a"):
        max_batches: int | None = None
        target_total: int | None = None
        rest = sys.argv[2:]
        if rest and rest[0] == "--target" and len(rest) > 1:
            target_total = int(rest[1])
        elif rest:
            max_batches = int(rest[0])
        action_auto_run(max_batches, target_total)
        return

    print("\nUsage:")
    print("  python scripts/generate_section_labeling_data.py --auto [N_BATCHES]")
    print("  python scripts/generate_section_labeling_data.py --auto --target 500")


if __name__ == "__main__":
    main()
