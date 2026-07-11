"""
Multi-provider LLM client with automatic key rotation and exhaustion tracking.

Reads API keys from the project .env file (via python-dotenv):
  DEEP_SEEK_KEY_1..N, GEMINI_API_KEY_1..N, GROQ_API_KEY_1..N, CEREBRAS_API_KEY_1..N,
  MISTRAL_API_KEY_1..N, MOONSHOT_API_KEY_1..N, ALIBABA_API_KEY_1..N, GPT_API_KEY_1..N,
  COHERE_API_KEY_1..N, OPENROUTER_API_KEY_1..N, SILICONFLOW_API_KEY_1..N,
  NVIDIA_API_KEY_1..N
(any number of trailing _N keys per provider is supported — add DEEP_SEEK_KEY_5
etc. and it's picked up automatically).

All keys are assumed to be free-tier with limited daily quota. call_llm() tries
every configured, non-exhausted key across all providers in a fixed priority
order (see PROVIDERS below) and returns the first successful response. A 429 /
auth-rejected response marks that specific key exhausted
(persisted in .api_key_status.json) so future calls skip it without retrying,
until the cooldown window (assumed daily reset) passes. Once every configured
key is exhausted or unusable, call_llm raises AllKeysExhaustedError so the
caller can stop cleanly instead of looping on failures forever.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

SCRIPTS_DIR  = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

STATUS_FILE = SCRIPTS_DIR / ".api_key_status.json"

# Free-tier quotas are typically daily; treat a 429/auth-rejected key as
# exhausted for this long before trying it again. Tune via env if a provider's
# actual reset window differs.
EXHAUSTION_COOLDOWN_HOURS = float(os.environ.get("LLM_KEY_COOLDOWN_HOURS", "20"))

# Deliberately separate from the app's general REQUEST_TIMEOUT_SECONDS (used
# for unrelated things like PDF OCR) — generating/scoring 20 pairs in one
# response is a large completion and 20-30s is routinely too short, causing
# spurious "timed out" fallbacks even when the provider would have answered.
TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "180"))


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    env_prefix: str      # e.g. "DEEP_SEEK_KEY_" -> DEEP_SEEK_KEY_1, _2, ...
    model_env: str        # env var to override the default model
    default_model: str
    style: str             # "openai" (chat/completions), "gemini", or "cohere"
    base_url_env: str      # env var to override the default base_url (wrong
                            # region/endpoint guesses shouldn't need a code edit)
    default_base_url: str
    max_tokens: int = 8192  # provider-specific output cap; only Cohere's
                             # command-r-plus-08-2024 hard-rejects >4096 (see below)
    timeout: float | None = None  # provider-specific override of TIMEOUT below;
                                    # None = use the global default


# Priority order: tried in this sequence, keys 1..N within each provider first.
# Default models/URLs are best-effort guesses — several providers have
# separate China-mainland vs international endpoints (Moonshot, Alibaba) or
# model catalogs that change often (Cerebras). A wrong guess just fails that
# provider's keys and falls through to the next, so it degrades gracefully —
# but override via the *_MODEL / *_BASE_URL env vars below if you know the
# correct value for your account (e.g. MOONSHOT_BASE_URL=https://api.moonshot.cn/v1/chat/completions).
PROVIDERS: list[ProviderSpec] = [
    ProviderSpec("deepseek", "DEEP_SEEK_KEY_", "DEEPSEEK_MODEL", "deepseek-chat", "openai",
                 "DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"),
    ProviderSpec("gemini", "GEMINI_API_KEY_", "GEMINI_MODEL", "gemini-2.0-flash", "gemini",
                 "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"),
    ProviderSpec("groq", "GROQ_API_KEY_", "GROQ_MODEL", "llama-3.3-70b-versatile", "openai",
                 "GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
    ProviderSpec("cerebras", "CEREBRAS_API_KEY_", "CEREBRAS_MODEL", "gpt-oss-120b", "openai",
                 "CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1/chat/completions"),
    ProviderSpec("mistral", "MISTRAL_API_KEY_", "MISTRAL_MODEL", "mistral-small-latest", "openai",
                 "MISTRAL_BASE_URL", "https://api.mistral.ai/v1/chat/completions"),
    ProviderSpec("moonshot", "MOONSHOT_API_KEY_", "MOONSHOT_MODEL", "moonshot-v1-8k", "openai",
                 "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1/chat/completions"),
    ProviderSpec("alibaba", "ALIBABA_API_KEY_", "ALIBABA_MODEL", "qwen-plus", "openai",
                 # Mainland China endpoint — confirmed correct for this account
                 # via its console showing region "China (Beijing)". Override
                 # ALIBABA_BASE_URL to the -intl variant if you switch to an
                 # international workspace later.
                 "ALIBABA_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ProviderSpec("gpt", "GPT_API_KEY_", "GPT_MODEL", "gpt-4o-mini", "openai",
                 "GPT_BASE_URL", "https://api.openai.com/v1/chat/completions"),
    ProviderSpec("cohere", "COHERE_API_KEY_", "COHERE_MODEL", "command-r-plus-08-2024", "cohere",
                 "COHERE_BASE_URL", "https://api.cohere.com/v2/chat", max_tokens=4096),
    ProviderSpec("openrouter", "OPENROUTER_API_KEY_", "OPENROUTER_MODEL",
                 "meta-llama/llama-3.3-70b-instruct:free", "openai",
                 "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"),
    ProviderSpec("siliconflow", "SILICONFLOW_API_KEY_", "SILICONFLOW_MODEL",
                 # International endpoint — confirmed correct via direct testing
                 # (the mainland .cn endpoint 401s for this account's keys).
                 "Qwen/Qwen2.5-7B-Instruct", "openai",
                 "SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1/chat/completions"),
    ProviderSpec("nvidia", "NVIDIA_API_KEY_", "NVIDIA_MODEL", "minimaxai/minimax-m3", "openai",
                 "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions",
                 # NVIDIA's free-tier hosted inference was observed hanging for
                 # 120s+ on real-size prompts (confirmed across multiple models,
                 # not just minimax-m3) or returning "DEGRADED function cannot
                 # be invoked" — a shorter timeout fails fast instead of
                 # burning ~180s per key when it's the last resort and also down.
                 timeout=60.0),
]

# Per-provider cap (see ProviderSpec.max_tokens above) — most providers default
# to 8192, which is what the 20-pair combined pair-scoring response actually
# needs (~4500-5500 tokens of verbose JSON; 4096 was found to truncate the
# last 1-2 pairs, causing spurious "got 19/18 lines" validation failures).
# Cohere's command-r-plus-08-2024 hard-rejects any max_tokens above 4096 with
# a 400 (confirmed by direct testing), so it keeps its own lower cap.
# Set LLM_MAX_OUTPUT_TOKENS to force the SAME value for every provider instead.
MAX_TOKENS_OVERRIDE = os.environ.get("LLM_MAX_OUTPUT_TOKENS")


class AllKeysExhaustedError(RuntimeError):
    """Raised when every configured API key is exhausted, invalid, or unusable."""


class QuotaExceeded(RuntimeError):
    """Raised internally when a single key hits a rate-limit/quota/auth error."""


def _discover_keys(spec: ProviderSpec) -> list[tuple[int, str]]:
    """All configured KEY_1, KEY_2, ... for this provider (stops at first gap)."""
    keys = []
    i = 1
    while True:
        val = os.environ.get(f"{spec.env_prefix}{i}")
        if val is None:
            break
        if val.strip():
            keys.append((i, val.strip()))
        i += 1
    return keys


def _load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_status(status: dict) -> None:
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _mark_exhausted(provider: str, key_index: int, reason: str) -> None:
    status = _load_status()
    status[f"{provider}:{key_index}"] = {
        "exhausted_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    _save_status(status)


def _is_exhausted(provider: str, key_index: int, status: dict) -> bool:
    entry = status.get(f"{provider}:{key_index}")
    if not entry:
        return False
    exhausted_at = datetime.fromisoformat(entry["exhausted_at"])
    return datetime.now(timezone.utc) - exhausted_at < timedelta(hours=EXHAUSTION_COOLDOWN_HOURS)


def iter_available_keys(exclude: set[str] | None = None):
    """Yield (spec, key_index, key_value) for every configured key that is
    neither excluded (already tried in this call_llm invocation) nor currently
    marked exhausted, in provider priority order."""
    exclude = exclude or set()
    status = _load_status()
    for spec in PROVIDERS:
        for key_index, key_value in _discover_keys(spec):
            token = f"{spec.name}:{key_index}"
            if token in exclude:
                continue
            if _is_exhausted(spec.name, key_index, status):
                continue
            yield spec, key_index, key_value


class MalformedResponse(RuntimeError):
    """Raised when a provider returns 200 OK but the body isn't usable text —
    e.g. a reasoning model (like Cerebras' gpt-oss-120b) that spent its whole
    max_tokens budget on internal reasoning and never emitted a final
    `content` string. Treated like any other per-key failure: skip, try next."""


def _extract_text_openai(data: dict) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise MalformedResponse(f"no choices[0].message in response: {e}") from e
    content = message.get("content")
    if content:
        return content
    # Reasoning-style models sometimes put output here instead of `content`
    # when content ends up empty (commonly because reasoning consumed the
    # entire token budget before any final-answer text was written).
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        return reasoning
    finish_reason = data["choices"][0].get("finish_reason")
    raise MalformedResponse(f"empty content (finish_reason={finish_reason!r}) — likely ran out of max_tokens")


def _extract_text_gemini(data: dict) -> str:
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _extract_text_cohere(data: dict) -> str:
    return "".join(part.get("text", "") for part in data["message"]["content"])


def _call_once(spec: ProviderSpec, key_value: str, prompt: str) -> str:
    model      = os.environ.get(spec.model_env, spec.default_model)
    base_url   = os.environ.get(spec.base_url_env, spec.default_base_url)
    max_tokens = int(MAX_TOKENS_OVERRIDE) if MAX_TOKENS_OVERRIDE else spec.max_tokens
    timeout    = spec.timeout if spec.timeout is not None else TIMEOUT

    if spec.style in ("openai", "cohere"):
        body = {
            "model":      model,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if spec.style == "openai":
            body["temperature"] = 0.9
        resp = httpx.post(
            base_url,
            headers={"Authorization": f"Bearer {key_value}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
    else:  # gemini
        resp = httpx.post(
            base_url.format(model=model),
            params={"key": key_value},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.9, "maxOutputTokens": max_tokens},
            },
            timeout=timeout,
        )

    if resp.status_code == 429:
        raise QuotaExceeded(f"rate limited / quota exceeded (HTTP 429)")
    if resp.status_code in (401, 402, 403):
        # 401/403 = key invalid or forbidden; 402 = billing/credit required
        # (seen from free-tier keys with no payment method on file). None of
        # these resolve by simply retrying the same key, so treat them the
        # same as a quota hit — skip this key for the cooldown window instead
        # of re-trying it (and failing) on every single future call.
        raise QuotaExceeded(f"key rejected (HTTP {resp.status_code})")
    resp.raise_for_status()

    data = resp.json()
    if spec.style == "openai":
        text = _extract_text_openai(data)
    elif spec.style == "cohere":
        text = _extract_text_cohere(data)
    else:
        text = _extract_text_gemini(data)
    return text.strip()


def call_llm(prompt: str, exclude: set[str] | None = None) -> tuple[str, str]:
    """Call the first available (non-exhausted, non-excluded) key.

    Returns (response_text, "<provider>:<key_index>") identifying which key
    produced the response — pass the accumulated set of used tokens back in
    as `exclude` on a retry to force a different key next time.

    Raises AllKeysExhaustedError once no usable key remains.
    """
    exclude = set(exclude or set())
    saw_any_key = False

    for spec, key_index, key_value in iter_available_keys(exclude):
        saw_any_key = True
        token = f"{spec.name}:{key_index}"
        try:
            text = _call_once(spec, key_value, prompt)
            return text, token
        except QuotaExceeded as e:
            print(f"  [llm] {token} exhausted ({e}) — trying next key")
            _mark_exhausted(spec.name, key_index, str(e))
            continue
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            print(f"  [llm] {token} request failed ({e}) — trying next key")
            continue
        except MalformedResponse as e:
            print(f"  [llm] {token} gave an unusable response ({e}) — trying next key")
            continue
        except Exception as e:
            # Catch-all: a provider we haven't seen a quirk from yet (unexpected
            # response shape, JSON decode failure, etc.) must never crash the
            # whole automation run — the entire point of this client is to
            # degrade to the next key/provider instead of stopping cold.
            print(f"  [llm] {token} raised unexpected {type(e).__name__} ({e}) — trying next key")
            continue

    if not saw_any_key:
        raise AllKeysExhaustedError(
            "No usable API keys available (all exhausted, excluded, or none configured in .env)."
        )
    raise AllKeysExhaustedError("All configured API keys failed on this request.")


def provider_tokens(token: str) -> set[str]:
    """All "<provider>:<index>" tokens sharing the same provider as `token`.

    Used by callers that got a well-formed-but-wrong response (e.g. a model
    that reliably outputs the wrong number of lines for a given prompt): a
    validation failure is a property of that PROVIDER's model, not of the
    specific key, so retrying with another key from the same provider just
    repeats the same mistake and wastes the retry budget before ever reaching
    a different provider lower in priority. Callers should add this whole set
    to their `exclude` set on a validation failure, not just the one token.
    """
    provider = token.split(":", 1)[0]
    for spec in PROVIDERS:
        if spec.name == provider:
            return {f"{provider}:{i}" for i, _ in _discover_keys(spec)}
    return {token}


VN_TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh — no DST, fixed UTC+7


def _fmt_local(dt: datetime) -> str:
    return dt.astimezone(VN_TZ).strftime("%Y-%m-%d %H:%M ICT")


def _is_account_issue(reason: str) -> bool:
    """True for 401/402/403-style rejections (invalid key, no billing/credit,
    forbidden) — these do NOT resolve just by waiting out the cooldown like a
    real 429 rate limit does. The cooldown still applies (so the key gets
    retried automatically rather than disabled forever), but it will very
    likely fail again with the same reason until you fix it on the
    provider's side (add credit, replace the key, etc.)."""
    return "402" in reason or "401" in reason or "403" in reason


def key_status_report() -> str:
    """Aggregate every configured key's availability into one human-readable
    report — when does each exhausted key clear its cooldown, and when does
    the WHOLE system (every configured provider) become usable again. Reads
    only the local .api_key_status.json + EXHAUSTION_COOLDOWN_HOURS, so it
    doesn't require checking each provider's dashboard/website by hand.
    Times are shown in Vietnam time (ICT, UTC+7)."""
    status = _load_status()
    now = datetime.now(timezone.utc)
    cooldown = timedelta(hours=EXHAUSTION_COOLDOWN_HOURS)

    lines: list[str] = []
    latest_clear: datetime | None = None
    any_configured = False
    any_available_now = False

    for spec in PROVIDERS:
        keys = _discover_keys(spec)
        if not keys:
            continue
        any_configured = True
        rows = []
        for key_index, _ in keys:
            token = f"{spec.name}:{key_index}"
            entry = status.get(token)
            if not entry:
                rows.append(f"    {token:<18} available now")
                any_available_now = True
                continue
            exhausted_at = datetime.fromisoformat(entry["exhausted_at"])
            clears_at = exhausted_at + cooldown
            account_issue = _is_account_issue(entry["reason"])
            note = "  ⚠ account issue, not a rate limit — likely fails again until you fix it (invalid key / no billing/credit)" if account_issue else ""
            if now >= clears_at:
                rows.append(f"    {token:<18} available now (cooldown ended){note}")
                any_available_now = True
            else:
                remaining = clears_at - now
                hrs, rem = divmod(int(remaining.total_seconds()), 3600)
                mins = rem // 60
                rows.append(
                    f"    {token:<18} exhausted ({entry['reason']}) — "
                    f"clears in {hrs}h{mins:02d}m, at {_fmt_local(clears_at)}{note}"
                )
                if latest_clear is None or clears_at > latest_clear:
                    latest_clear = clears_at
        lines.append(f"  {spec.name}:")
        lines.extend(rows)

    if not any_configured:
        return "  No API keys configured in .env."

    header = "  === API key cooldown status (times in Vietnam/ICT, UTC+7) ===\n"
    footer_parts = []
    if any_available_now:
        footer_parts.append("  → At least one key is usable right now.")
    else:
        footer_parts.append("  → No key is usable right now.")
    if latest_clear is not None:
        remaining = latest_clear - now
        hrs, rem = divmod(max(int(remaining.total_seconds()), 0), 3600)
        mins = rem // 60
        footer_parts.append(
            f"  → ALL configured keys will be usable by {_fmt_local(latest_clear)} "
            f"(in {hrs}h{mins:02d}m — the last one to clear)."
        )
    else:
        footer_parts.append("  → All keys are currently available (nothing exhausted).")
    return header + "\n".join(lines) + "\n\n" + "\n".join(footer_parts) + "\n"


if __name__ == "__main__":
    print(key_status_report())
