"""
Synthetic section-labeling data generator for the JD section-splitter
distillation project — mirrors scripts/generate_section_labeling_data.py
(the CV version) but for job descriptions.

Like the CV generator, this is the OPPOSITE of scripts/generate_synthetic_data.py's
JD prompt: that script forbids bullet points and section headers (prose-only,
for CV-JD scoring). This script REQUIRES them — headers, bullets, varied
layout, occasional no-header JDs — because the student model being trained
here (training/fine_tune_jd_section_classifier.py) needs to learn header/
bullet/layout boundary cues that app/parsers/job_description_parser.py's
fixed alias list cannot generalize to.

Each LLM call asks for a small batch of JDs, each pre-assigned a
header_style / language / hard_case / domain (rotated for diversity), and
gets back one JSON object per JD: the JD's lines, each labeled with the
section it belongs to (see app.parsers.job_description_parser.SECTION_ORDER
for the label vocabulary — imported directly so it can never drift from the
splitter's contract).

Retries only re-request whatever JDs are still missing/invalid from a batch
(same accumulate-and-retry-only-missing pattern as the CV generator),
resumable via progress already written to disk (no separate session file
needed — see scripts/generate_section_labeling_data.py's note on why).

Run:
  python scripts/generate_jd_labeling_data.py --auto [N_BATCHES]
  python scripts/generate_jd_labeling_data.py --auto --target 500
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

from app.parsers.job_description_parser import SECTION_ORDER  # noqa: E402

OUTPUT_DIR       = PROJECT_ROOT / "datasets" / "jd_section_splitting" / "v0.1"
OUTPUT_PATH      = OUTPUT_DIR / "labeled_lines.jsonl"
STYLE_USAGE_FILE = SCRIPTS_DIR / ".jd_section_style_usage.json"
TEMP_INPUT       = SCRIPTS_DIR / "jd_section_labeling_temp_input.txt"
PROMPT_OUTPUT    = SCRIPTS_DIR / "jd_section_labeling_prompt_output.txt"

# ── constants ─────────────────────────────────────────────────────────────────

ALLOWED_LABELS = set(SECTION_ORDER) | {"other"}

BATCH_SIZE = 3  # JDs per LLM call — same reasoning as the CV generator: each
                # JD's output is a full line-by-line JSON array, verbose per item.

MAX_ATTEMPTS_PER_STEP = 12
PROVIDER_STRIKES_BEFORE_SKIP = 2

MIN_LINES_PER_JD    = 12
MIN_DISTINCT_LABELS = 3  # out of 5 total — a short JD may legitimately skip
                          # "benefits" or "nice_to_have" entirely.

HEADER_STYLES = [
    "ALL CAPS headers on their own line (e.g. \"RESPONSIBILITIES\")",
    "Title Case headers on their own line (e.g. \"Key Responsibilities\")",
    "Markdown-bold headers (e.g. \"**Requirements**\")",
    "inline label-colon style with no standalone header line (e.g. \"Requirements: 3+ years Python, FastAPI, PostgreSQL\")",
    "minimal-to-no explicit headers — sections must be distinguishable mainly from content/context, not header text",
]

LANGUAGES = ["en", "vi", "mixed"]

HARD_CASES = [
    "none - straightforward single-column layout",
    "stacked bullets - list all Responsibilities and Requirements bullets first, then a trailing Nice-to-have block bunched at the very end (simulates a PDF-extraction column artifact)",
    "wrapped requirement line - a single requirement wraps across 2-3 separate lines with no punctuation cue that they're a continuation",
    "two-column artifact - interleave Responsibilities bullets and Requirements bullets as if a 2-column PDF layout collapsed into one column",
]

DOMAINS = [
    "Backend Engineering", "Frontend Engineering", "Mobile Development", "Data Engineering",
    "DevOps", "QA/Test Automation", "Product Management", "UI/UX Design", "Digital Marketing",
    "Sales", "Customer Support", "Human Resources", "Finance", "Business Analysis",
    "Content Writing", "Supply Chain", "Legal", "Healthcare Administration",
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE   = re.compile(r"https?://\S+", re.I)

LABEL_LEGEND = """  responsibilities   day-to-day duties, what the role actually does
  requirements       must-have qualifications, required skills/experience
  nice_to_have       preferred/bonus qualifications, "a plus" items
  benefits           perks, compensation notes, why-join-us content
  other              use ONLY for lines with no clear section membership
                     (e.g. the job title itself, a company/team intro
                     paragraph, a decorative separator). Do NOT use "other"
                     for content that clearly belongs to a section just
                     because you're unsure of exact wording."""

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
    counts = usage.setdefault(axis, {})
    min_count = min((counts.get(o, 0) for o in options), default=0)
    least_used = [o for o in options if counts.get(o, 0) == min_count]
    chosen = random.choice(least_used)
    counts[chosen] = counts.get(chosen, 0) + 1
    return chosen

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
    print(f"  Labeled JDs: {len(records)}   Total labeled lines: {total_lines}")
    if total_lines:
        print("  Label distribution (lines):")
        for label in sorted(ALLOWED_LABELS):
            count = label_counts.get(label, 0)
            pct = count / total_lines * 100
            print(f"    {label:<16} {count:>6}  ({pct:4.1f}%)")
    print("=" * 60)

# ── prompt builder ─────────────────────────────────────────────────────────────

def _next_start_id() -> int:
    return get_max_numeric_id(load_jsonl(OUTPUT_PATH), "jdline_") + 1


def _build_batch_specs(start_id: int, batch_size: int = BATCH_SIZE) -> list[dict]:
    usage = _load_usage()
    specs = []
    for i in range(batch_size):
        specs.append({
            "id":           f"jdline_{start_id + i:04d}",
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
A previous response was incomplete. Output ONLY the job descriptions listed
below — do not re-output or repeat any other one.
"""

    return f"""You are generating training data to teach a small model how real
job description documents are laid out into sections. This is DIFFERENT
from other JD generation you may have done before: those wanted clean prose
with NO headers or bullets. This task wants the OPPOSITE — realistic JD
documents with EXPLICIT section headers and bullet points (except where a
JD's assignment below calls for a minimal/no-header style), because the
model being trained needs real header/bullet/layout cues to learn from.
{partial_note}
════════════════════════════════════
JOB DESCRIPTIONS TO GENERATE ({len(specs)})
════════════════════════════════════
{assignment_lines}

For each JD: write a realistic job posting for the given domain (any
seniority — vary it). Do NOT include a company name anywhere in the text.

════════════════════════════════════
LAYOUT REQUIREMENTS PER header_style
════════════════════════════════════
- "ALL CAPS headers": each section starts with its name in ALL CAPS on its
  own line (e.g. "RESPONSIBILITIES"), followed by bullet-point or short-line content.
- "Title Case headers": same but Title Case (e.g. "Key Responsibilities").
- "Markdown-bold headers": headers wrapped in ** (e.g. "**Requirements**").
- "inline label-colon style": no standalone header line at all — each
  section's content starts right after "Label: " on the same line.
- "minimal-to-no explicit headers": write the JD as a sequence of clearly
  distinguishable content blocks (a requirement line still reads like a
  requirement, a responsibility bullet still reads like a duty) but WITHOUT
  a labeled header line introducing most sections.
Apply the hard_case for each JD literally where it's not "none" — these are
deliberately awkward/realistic layout quirks the model must learn to
handle, do not smooth them over.

════════════════════════════════════
LINE-LEVEL LABELING — every line gets exactly one label
════════════════════════════════════
{LABEL_LEGEND}

A header line itself (e.g. the line that just says "REQUIREMENTS") gets the
SAME label as the section it introduces — do not invent a separate "header"
label. The job title line and any intro/team-description paragraph before
the first section are "other". Bullet lines keep their bullet marker (e.g.
"- Design REST APIs...") in `text`. Blank/empty lines should simply be
omitted from the `lines` array (don't emit empty-string entries).

Each JD needs at least {MIN_LINES_PER_JD} lines total and at least
{MIN_DISTINCT_LABELS} distinct section labels represented.

════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════
Output exactly {len(specs)} lines. Each line is a single JSON object (no
array brackets, no commas between lines, no markdown fences, no explanation).

Schema per JD:
{{"id":"<jdline_id>","lines":[{{"text":"<line text, bullet marker included if present>","label":"<responsibilities|requirements|nice_to_have|benefits|other>"}}, ...]}}
"""

# ── validation ─────────────────────────────────────────────────────────────────

def validate_labeled_jd(obj: dict) -> list[str]:
    jid = obj.get("id", "?")
    errs: list[str] = []

    for field in ("id", "lines"):
        if field not in obj:
            errs.append(f"JD {jid}: missing field '{field}'")

    lines = obj.get("lines")
    if not isinstance(lines, list) or not lines:
        errs.append(f"JD {jid}: 'lines' must be a non-empty list")
        return errs

    if len(lines) < MIN_LINES_PER_JD:
        errs.append(f"JD {jid}: only {len(lines)} lines, minimum {MIN_LINES_PER_JD}")

    labels_seen: set[str] = set()
    full_text_parts: list[str] = []
    for i, item in enumerate(lines, 1):
        if not isinstance(item, dict) or "text" not in item or "label" not in item:
            errs.append(f"JD {jid} line {i}: must be an object with 'text' and 'label'")
            continue
        if not isinstance(item["text"], str) or not item["text"].strip():
            errs.append(f"JD {jid} line {i}: 'text' must be a non-empty string")
        if item.get("label") not in ALLOWED_LABELS:
            errs.append(f"JD {jid} line {i}: invalid label '{item.get('label')}'")
        else:
            labels_seen.add(item["label"])
        full_text_parts.append(str(item.get("text", "")))

    if len(labels_seen) < MIN_DISTINCT_LABELS:
        errs.append(
            f"JD {jid}: only {len(labels_seen)} distinct section label(s) used "
            f"({sorted(labels_seen)}), minimum {MIN_DISTINCT_LABELS}"
        )

    joined = "\n".join(full_text_parts)
    if EMAIL_RE.search(joined):
        errs.append(f"JD {jid}: contains an email-like pattern")
    if URL_RE.search(joined):
        errs.append(f"JD {jid}: contains a URL pattern")

    return errs


def _process_labeling_lines(
    lines: list[str], expected_ids: set[str],
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Parse + validate labeling-response lines against `expected_ids`.
    Mirrors scripts/generate_section_labeling_data.py's function of the same
    name: returns (accepted, errors, warnings) where `accepted` only
    contains ids that both parsed and passed validation in THIS response."""
    parse_errors: list[str] = []
    parsed: list[dict] = []
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            parse_errors.append(f"Line {i}: invalid JSON — {e}")
            continue
        if not isinstance(obj, dict):
            parse_errors.append(f"Line {i}: expected a JSON object, got {type(obj).__name__}")
            continue
        parsed.append(obj)

    warnings: list[str] = []
    by_id: dict[str, dict] = {}
    for obj in parsed:
        jid = obj.get("id")
        if jid not in expected_ids:
            warnings.append(f"[ID] JD '{jid}' is unexpected/extra — dropped")
            continue
        if jid in by_id:
            warnings.append(f"[ID] JD '{jid}' is duplicated — keeping first occurrence")
            continue
        by_id[jid] = obj

    errors: list[str] = list(parse_errors)
    accepted: dict[str, dict] = {}
    for jid, obj in by_id.items():
        errs = validate_labeled_jd(obj)
        if errs:
            errors.extend(errs)
            continue
        accepted[jid] = obj

    for jid in sorted(expected_ids - set(by_id)):
        errors.append(f"[ID] JD '{jid}' expected but missing from LLM output")

    return accepted, errors, warnings

# ── fully-automated mode (calls .env API keys directly) ───────────────────────

try:
    import llm_client
    _LLM_CLIENT_AVAILABLE = True
except ImportError:
    _LLM_CLIENT_AVAILABLE = False


def _record_provider_failure(used_key: str, provider_fails: dict[str, int], tried: set[str]) -> None:
    provider = used_key.split(":", 1)[0]
    provider_fails[provider] = provider_fails.get(provider, 0) + 1
    if provider_fails[provider] >= PROVIDER_STRIKES_BEFORE_SKIP:
        tried |= llm_client.provider_tokens(used_key)


def _auto_generate_batch(specs: list[dict]) -> list[dict]:
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
                  f"+{len(accepted)} JD(s) accepted ({len(collected)}/{len(specs)} total)")
        if len(accepted) < len(missing_ids):
            still_missing = len(missing_ids) - len(accepted)
            print(f"  [auto]   {still_missing} JD(s) still missing/invalid:")
            for e in errors[:5]:
                print(f"      ✗ {e}")
            if not accepted:
                _record_provider_failure(used_key, provider_fails, tried)

    missing_ids = expected_ids - set(collected)
    if missing_ids:
        raise RuntimeError(
            f"Batch: still missing {len(missing_ids)}/{len(specs)} JD(s) after "
            f"{MAX_ATTEMPTS_PER_STEP} attempts across different keys: {sorted(missing_ids)}"
        )

    return [collected[s["id"]] for s in specs]


def _auto_run_one_batch() -> None:
    start_id = _next_start_id()
    specs = _build_batch_specs(start_id)
    print(f"\n  [auto] generating batch: {specs[0]['id']}..{specs[-1]['id']}")
    for s in specs:
        print(f"    {s['id']}: {s['domain']} | {s['header_style'][:40]}... | {s['language']} | {s['hard_case'][:30]}...")

    jds_new = _auto_generate_batch(specs)
    append_jsonl(OUTPUT_PATH, jds_new)
    total_lines = sum(len(j["lines"]) for j in jds_new)
    print(f"  [auto] ✓ saved {len(jds_new)} JD(s), {total_lines} labeled lines → {OUTPUT_PATH.name}")

    auto_commit_batch()


def auto_commit_batch() -> None:
    """Stage and commit this batch's files, scoped to exactly what this
    script writes (not `git add .`)."""
    files = [OUTPUT_PATH, TEMP_INPUT, PROMPT_OUTPUT, STYLE_USAGE_FILE]
    try:
        subprocess.run(
            ["git", "add", "--", *[str(f) for f in files]],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", "feat: add JD section-labeling training data batch"],
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


def action_auto_run(max_batches: int | None, target_total: int | None) -> None:
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
                print(f"\n  [auto] target of {target_total} JDs reached.")
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
    print("║   JD Section-Labeling Data Generator (distillation)  ║")
    print(f"║   batch size: {BATCH_SIZE} JDs/call                             ║")
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
    print("  python scripts/generate_jd_labeling_data.py --auto [N_BATCHES]")
    print("  python scripts/generate_jd_labeling_data.py --auto --target 500")


if __name__ == "__main__":
    main()
