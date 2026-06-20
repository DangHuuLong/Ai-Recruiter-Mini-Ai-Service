"""
Call LLM API on each relabeling prompt file and save JSON responses.

Supports Gemini (free tier), Groq (free tier), Cerebras (free tier), and Anthropic (paid).
Default provider is Gemini - free with GEMINI_API_KEY.

Prerequisites:
  Gemini (free tier, recommended):
    pip install google-genai
    Add to .env: GEMINI_API_KEY=AIza...
    Get key at: https://aistudio.google.com/apikey

  Groq (free tier, fast):
    pip install groq
    Add to .env: GROQ_API_KEY=gsk_...
    Get key at: https://console.groq.com/keys

  Cerebras (free tier, high TPM):
    pip install cerebras-cloud-sdk
    Add to .env: CEREBRAS_API_KEY=csk_...
    Get key at: https://cloud.cerebras.ai

  Anthropic (paid, ~$2 for all 421 batches with Haiku):
    pip install anthropic
    Add to .env: ANTHROPIC_API_KEY=sk-ant-...

Usage:
  python scripts/run_relabeling.py                         # Gemini, all zones
  python scripts/run_relabeling.py --provider groq         # Groq free tier
  python scripts/run_relabeling.py --provider cerebras     # Cerebras free tier
  python scripts/run_relabeling.py --provider anthropic
  python scripts/run_relabeling.py --zone weak_moderate_60
  python scripts/run_relabeling.py --batch-limit 3         # test 3 files first
  python scripts/run_relabeling.py --dry-run               # preview only
  python scripts/run_relabeling.py --no-resume             # reprocess all
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

PROMPTS_DIR   = Path("artifacts/relabeling/prompts")
RESPONSES_DIR = Path("artifacts/relabeling/responses")

GEMINI_DEFAULT_MODEL    = "gemini-2.5-flash"
GROQ_DEFAULT_MODEL      = "llama-3.3-70b-versatile"
CEREBRAS_DEFAULT_MODEL  = "gpt-oss-120b"
ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Gemini free tier: 15 RPM -> need 4s gap minimum
# Groq free tier: 30 RPM -> need 2s gap minimum
# Cerebras free tier: 30 RPM, 60K TPM -> need 2s gap minimum
GEMINI_FREE_DELAY    = 4.5
GROQ_FREE_DELAY      = 2.5
CEREBRAS_FREE_DELAY  = 2.5
ANTHROPIC_PAID_DELAY = 0.5


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def parse_json_blocks(text: str) -> list[dict]:
    """Extract all ```json ... ``` blocks from LLM response text."""
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for m in matches:
        try:
            results.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass
    return results


def parse_bare_json_objects(text: str) -> list[dict]:
    """Fallback: scan for { ... } blocks that contain pair_id."""
    results = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = text[start : i + 1]
                try:
                    obj = json.loads(fragment)
                    if "pair_id" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return results


def extract_relabels(response_text: str) -> list[dict]:
    results = parse_json_blocks(response_text)
    if not results:
        results = parse_bare_json_objects(response_text)
    return results


# ---------------------------------------------------------------------------
# Provider: Gemini
# ---------------------------------------------------------------------------

def load_gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key or None


def make_gemini_client(api_key: str):
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        raise SystemExit(
            "google-genai package not installed.\n"
            "Run: .venv\\Scripts\\pip.exe install google-genai"
        )


def call_gemini(client, prompt_text: str, model: str) -> str:
    from google import genai
    response = client.models.generate_content(
        model=model,
        contents=prompt_text,
    )
    return response.text


# ---------------------------------------------------------------------------
# Provider: Groq
# ---------------------------------------------------------------------------

def load_groq_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key or None


def make_groq_client(api_key: str):
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        raise SystemExit(
            "groq package not installed.\n"
            "Run: .venv\\Scripts\\pip.exe install groq"
        )


def call_groq(client, prompt_text: str, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Provider: Cerebras
# ---------------------------------------------------------------------------

def load_cerebras_key() -> str | None:
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("CEREBRAS_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key or None


def make_cerebras_client(api_key: str):
    try:
        from cerebras.cloud.sdk import Cerebras
        return Cerebras(api_key=api_key)
    except ImportError:
        raise SystemExit(
            "cerebras-cloud-sdk package not installed.\n"
            "Run: .venv\\Scripts\\pip.exe install cerebras-cloud-sdk"
        )


def call_cerebras(client, prompt_text: str, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=16384,
    )
    content = response.choices[0].message.content
    if content is None:
        finish_reason = response.choices[0].finish_reason
        raise ValueError(f"Cerebras returned empty content (finish_reason={finish_reason})")
    return content


# ---------------------------------------------------------------------------
# Provider: Anthropic
# ---------------------------------------------------------------------------

def load_anthropic_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key or None


def make_anthropic_client(api_key: str):
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise SystemExit(
            "anthropic package not installed.\n"
            "Run: .venv\\Scripts\\pip.exe install anthropic"
        )


def call_anthropic(client, prompt_text: str, model: str) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run LLM relabeling on prompt batch files")
    parser.add_argument("--provider",    default="gemini", choices=["gemini", "groq", "cerebras", "anthropic"],
                        help="LLM provider: gemini (free), groq (free), cerebras (free) or anthropic (paid)")
    parser.add_argument("--model",       default=None,
                        help="Model override (default: gemini-2.0-flash or claude-haiku-4-5-20251001)")
    parser.add_argument("--zone",        default=None,
                        help="Only process one zone, e.g. weak_moderate_60")
    parser.add_argument("--batch-limit", type=int, default=None,
                        help="Process only first N batch files (for testing)")
    parser.add_argument("--resume",      action="store_true", default=True,
                        help="Skip already-processed files (default: on)")
    parser.add_argument("--no-resume",   dest="resume", action="store_false",
                        help="Reprocess all files")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show plan without calling API")
    parser.add_argument("--delay",       type=float, default=None,
                        help="Seconds between API calls (default: 4.5 for Gemini, 0.5 for Anthropic)")
    args = parser.parse_args()

    # Resolve model and delay defaults per provider
    if args.provider == "gemini":
        model = args.model or GEMINI_DEFAULT_MODEL
        delay = args.delay if args.delay is not None else GEMINI_FREE_DELAY
    elif args.provider == "groq":
        model = args.model or GROQ_DEFAULT_MODEL
        delay = args.delay if args.delay is not None else GROQ_FREE_DELAY
    elif args.provider == "cerebras":
        model = args.model or CEREBRAS_DEFAULT_MODEL
        delay = args.delay if args.delay is not None else CEREBRAS_FREE_DELAY
    else:
        model = args.model or ANTHROPIC_DEFAULT_MODEL
        delay = args.delay if args.delay is not None else ANTHROPIC_PAID_DELAY

    # Collect prompt files (needed even for dry-run)
    prompt_files = sorted(PROMPTS_DIR.glob("relabel_*.txt"))
    if args.zone:
        prompt_files = [f for f in prompt_files if args.zone in f.name]
    if args.batch_limit:
        prompt_files = prompt_files[: args.batch_limit]

    if not prompt_files:
        raise SystemExit(f"No prompt files found in {PROMPTS_DIR} (zone={args.zone})")

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    to_process = []
    for pf in prompt_files:
        rf = RESPONSES_DIR / pf.with_suffix(".json").name
        if args.resume and rf.exists():
            continue
        to_process.append(pf)

    done_count = len(prompt_files) - len(to_process)
    eta_minutes = len(to_process) * delay / 60

    print(f"Provider     : {args.provider}")
    print(f"Model        : {model}")
    print(f"Prompt files : {len(prompt_files)}  |  done: {done_count}  |  to process: {len(to_process)}")
    print(f"Delay        : {delay}s/call  ->  ETA ~{eta_minutes:.0f} min")

    if args.dry_run:
        print("(dry-run - no API calls)")
        for pf in to_process[:10]:
            print(f"  would process: {pf.name}")
        if len(to_process) > 10:
            print(f"  ... and {len(to_process) - 10} more")
        return

    # Setup client
    if args.provider == "gemini":
        api_key = load_gemini_key()
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY not found.\n"
                "Get a free key at https://aistudio.google.com/apikey\n"
                "Then add to .env: GEMINI_API_KEY=AIza..."
            )
        client  = make_gemini_client(api_key)
        call_fn = lambda c, p, m: call_gemini(c, p, m)
    elif args.provider == "groq":
        api_key = load_groq_key()
        if not api_key:
            raise SystemExit(
                "GROQ_API_KEY not found.\n"
                "Get a free key at https://console.groq.com/keys\n"
                "Then add to .env: GROQ_API_KEY=gsk_..."
            )
        client  = make_groq_client(api_key)
        call_fn = lambda c, p, m: call_groq(c, p, m)
    elif args.provider == "cerebras":
        api_key = load_cerebras_key()
        if not api_key:
            raise SystemExit(
                "CEREBRAS_API_KEY not found.\n"
                "Get a free key at https://cloud.cerebras.ai\n"
                "Then add to .env: CEREBRAS_API_KEY=csk_..."
            )
        client  = make_cerebras_client(api_key)
        call_fn = lambda c, p, m: call_cerebras(c, p, m)
    else:
        api_key = load_anthropic_key()
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY not found.\n"
                "Add to .env: ANTHROPIC_API_KEY=sk-ant-..."
            )
        client  = make_anthropic_client(api_key)
        call_fn = lambda c, p, m: call_anthropic(c, p, m)

    processed   = 0
    errors      = 0
    total_pairs = 0

    for idx, prompt_file in enumerate(to_process, start=1):
        response_file = RESPONSES_DIR / prompt_file.with_suffix(".json").name

        header_line = prompt_file.read_text(encoding="utf-8").split("\n")[1]
        m = re.search(r"Pairs \((\d+)\)", header_line)
        expected_count = int(m.group(1)) if m else 10

        print(f"[{idx}/{len(to_process)}] {prompt_file.name} ({expected_count} pairs)", end=" ... ", flush=True)

        try:
            prompt_text   = prompt_file.read_text(encoding="utf-8")
            response_text = call_fn(client, prompt_text, model)
            relabels      = extract_relabels(response_text)

            result = {
                "source_file":  prompt_file.name,
                "provider":     args.provider,
                "model":        model,
                "pair_count":   expected_count,
                "parsed_count": len(relabels),
                "raw_response": response_text,
                "relabels":     relabels,
            }
            response_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total_pairs += len(relabels)
            processed   += 1
            status = "OK" if len(relabels) == expected_count else f"WARN {len(relabels)}/{expected_count} parsed"
            print(status)

        except Exception as exc:
            errors += 1
            print(f"ERROR: {exc}")
            (RESPONSES_DIR / (prompt_file.stem + "_error.txt")).write_text(str(exc), encoding="utf-8")

        if idx < len(to_process):
            time.sleep(delay)

    print(f"\nDone: {processed} processed, {done_count} skipped, {errors} errors")
    print(f"Total pairs relabeled: {total_pairs}")
    if errors:
        print(f"Re-run with --no-resume to retry errors, or check *_error.txt in {RESPONSES_DIR}")
    if processed:
        print(f"\nNext: python scripts/apply_relabels.py --dry-run")


if __name__ == "__main__":
    main()
