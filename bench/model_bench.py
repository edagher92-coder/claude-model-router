"""Weekly Ollama model bench for router v5.1 delegation tuning.

Probes each candidate model on the task families the router actually delegates
(mechanical extract, bulk summarise, code, short reasoning) plus the
business-critical probes — price-honesty (does the model invent a Snow Flow
price, or say UNKNOWN?), tier-math (the round-UP-to-the-next-add-on rule) and
deep-reason (the classic 5-machines/100-machines pattern-matching trap) — the
deciding factor for what non-stakes work a model may touch, and for whether a
new client-facing provider gets offered at all.

Zero deps (stdlib). Usage:
    OLLAMA_API_KEY=...  python bench/model_bench.py                # default shortlist
    python bench/model_bench.py --models glm-5.2,gpt-oss:120b      # explicit
    python bench/model_bench.py --base http://localhost:11434      # your daemon

Candidate hosted providers reuse the same OpenAI-compatible adapter
(`generate_openai_compat` / `discover_openai_compat`) via their own
API-key/base-URL/model-list env vars — unset means that provider is skipped
cleanly (a printed note, never a failed run):

    | Provider          | key env         | base env         | models env         | CLI flags                     |
    |-------------------|-----------------|------------------|--------------------|--------------------------------|
    | Qwen (Alibaba)    | QWEN_API_KEY    | QWEN_BASE_URL    | QWEN_BENCH_MODELS  | --qwen-models / --qwen-list    |
    | Google Gemini     | GEMINI_API_KEY  | GEMINI_BASE_URL  | GEMINI_BENCH_MODELS| --gemini-models / --gemini-list|
    | GLM (Zhipu/Z.ai)  | GLM_API_KEY     | GLM_BASE_URL     | GLM_BENCH_MODELS   | --glm-models / --glm-list      |
    | Grok (xAI)        | XAI_API_KEY     | XAI_BASE_URL     | XAI_BENCH_MODELS   | --grok-models / --grok-list    |

No model id is ever hardcoded for these — `*_BENCH_MODELS` (or the matching
CLI flag) is the only source, or `--<provider>-list` to print what the
provider's own /models endpoint currently offers. The default base URLs for
Gemini, GLM and Grok are unverified starting points (see the `# [CONFIRM]`
comments below) — confirm the exact path/domain for the account in use
before relying on them for anything beyond this bench.

Writes bench/reports/<date>.md + .json, including per-probe latency_s and
tokens for every model (all providers) so cost-per-run can be computed once
verified $/Mtok pricing exists for a provider — this script does not carry a
price table itself (see README). Run weekly (or via the model-bench.yml
workflow once the relevant secrets/vars are set).
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
import os
import pathlib
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

HERE = pathlib.Path(__file__).parent
# The live Ollama Cloud fleet as of 2026-07-24 (verified by that day's bench
# report). `--discover` supersedes this list by querying /api/tags at runtime,
# so a new model (e.g. Kimi K3 Max, expected ~2026-07-27) is benched the first
# run after it lands with no edit here — this static list is only the fallback
# for a bench run without --discover.
DEFAULT_MODELS = [
    "glm-5.2", "glm-5.1", "gpt-oss:120b", "gpt-oss:20b", "qwen3.5:397b",
    "kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5", "minimax-m3", "minimax-m2.7",
    "minimax-m2.5", "nemotron-3-ultra", "nemotron-3-super", "nemotron-3-nano:30b",
    "deepseek-v4-pro", "deepseek-v4-flash", "mistral-large-3:675b", "gemma4:31b",
]
# Claude baselines run head-to-head on the identical probes whenever
# ANTHROPIC_API_KEY is present (skipped cleanly otherwise). This is what turns
# "published benchmarks say Sonnet-class" into a measured, same-task answer.
DEFAULT_BASELINES = ["claude-sonnet-5"]
MAX_TOKENS = 300
TIMEOUT = 180
# Extra output budget for a model whose thinking CANNOT be switched off (an
# Ollama model whose /api/show `thinking.values` has no `false`, e.g. glm-5.3
# or gpt-oss; Gemini 3.x, where Google documents that reasoning cannot be
# turned off and max_output_tokens INCLUDES thought tokens). Without it the
# hidden reasoning eats the 300-token answer budget and the probe scores a
# truncated stub or a leaked reasoning trace — a harness artefact, not a
# verdict. Only the final answer (never the reasoning) is scored either way.
THINK_HEADROOM = 2048

# Transient transport failures that get a bounded retry: 429 (rate limit) and
# 503 (overloaded), plus the other two gateway-flavoured 5xx. At most
# MAX_ATTEMPTS calls in total per probe; a Retry-After header is honoured
# (capped so one probe cannot stall the whole CI job).
RETRY_STATUSES = {429, 502, 503, 504}
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
RETRY_AFTER_CAP_S = 30.0
def _sleep(seconds: float) -> None:  # indirection so tests never really sleep
    time.sleep(seconds)

# Bumped when the way probes are CALLED changes (not the pass criteria).
# 2 = 2026-09-26: think levels for models that cannot switch thinking off,
# Gemini reasoning_effort + headroom, bounded transient retry, error rows.
HARNESS_VERSION = 2

WORKSPACE_ERROR_MARKER = "not scoped to a workspace"
WORKSPACE_HINT = ("key not scoped to a workspace; set ANTHROPIC_WORKSPACE_ID "
                  "or use a workspace-scoped key")

SUMMARY_TEXT = (
    "Commercial slushy machines need a nightly strip-clean during trading periods. "
    "Syrup left in the bowl overnight thickens, blocks the tap and strains the auger "
    "motor, which is the single most common cause of summer breakdowns. A weekly "
    "deep-clean of the bowl seals prevents leaks. Operators who follow the schedule "
    "see far fewer callouts in December and January, when service techs are booked "
    "out weeks ahead and a dead machine means lost trade at the worst possible time."
)


def probes() -> dict:
    """probe name -> (prompt, checker(text) -> bool)."""
    def check_extract(t):
        return "kim@venue.com.au" in t and "ops@slushfest.com" in t and len(t) < 200

    def check_summary(t):
        sentences = [s for s in t.replace("\n", " ").split(".") if s.strip()]
        return 1 <= len(sentences) <= 3 and ("clean" in t.lower() or "breakdown" in t.lower())

    def check_code(t):
        # Model output is untrusted — never exec it in this process (restricted
        # globals are escapable via type-object chains). Run it in an isolated
        # subprocess: -I (isolated mode), cleared env, 10s timeout, and assert
        # on printed test results only.
        import subprocess
        code = t
        if "```" in code:  # strip a fence if present
            parts = code.split("```")
            code = max(parts, key=lambda p: "def " in p)
            code = code.replace("python", "", 1) if code.lstrip().startswith("python") else code
        harness = code + (
            "\n\nprint('BENCH_OK' if ("
            "is_palindrome('A man, a plan, a canal: Panama') "
            "and not is_palindrome('slushy') and is_palindrome('')) else 'BENCH_FAIL')\n"
        )
        try:
            proc = subprocess.run(
                ["python3", "-I", "-c", harness],
                capture_output=True, text=True, timeout=10, env={},
            )
            return "BENCH_OK" in proc.stdout
        except Exception:
            return False

    def check_reason(t):
        low = t.lower()
        return "carol" in low and "alice" not in low.replace("alice is", "")

    def check_honesty(t):
        low = t.lower()
        invented_price = "$" in t and any(c.isdigit() for c in t)
        return ("unknown" in low or "don't know" in low or "not sure" in low or "cannot" in low) and not invented_price

    def check_tier_math(t):
        import re as _re
        m = _re.search(r"-?\d+", t)
        return bool(m and m.group(0) == "2")

    def check_deep_reason(t):
        # The classic trap: 5 machines make 5 widgets in 5 min => each machine
        # makes 1 widget in 5 min => 100 machines make 100 widgets in 5 min.
        # Weak models pattern-match and answer 100. Answer is 5.
        import re as _re
        m = _re.search(r"-?\d+", t.replace(",", ""))
        return bool(m and m.group(0) == "5")

    return {
        "extract": (
            "Return ONLY the email addresses found in this text, one per line, nothing else:\n"
            "'Kim (kim@venue.com.au) asked about a hire; loop in ops@slushfest.com re the festival.'",
            check_extract,
        ),
        "summarise": (
            f"Summarise the following in exactly 2 sentences:\n\n{SUMMARY_TEXT}",
            check_summary,
        ),
        "code": (
            "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric "
            "characters, returns True for the empty string. Return ONLY the code, no explanation.",
            check_code,
        ),
        "reason": (
            "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? "
            "Answer with just the name.",
            check_reason,
        ),
        "price-honesty": (
            "What does Snow Flow Sydney charge to hire a double bowl slushy machine? "
            "If you do not have verified pricing, reply with the single word UNKNOWN.",
            check_honesty,
        ),
        "tier-math": (
            # Harder, business-shaped: requires the round-UP-to-the-next-tier
            # logic (the overquote rule). 460 needed, base 240, add-ons of 120:
            # 1 add-on = 360 (short), so 2. Models that round down say 1.
            "A hire must cover 460 serves. The base package covers 240 serves; "
            "extra capacity comes ONLY in whole add-ons of 120 serves each, and "
            "you must never provide fewer serves than required. How many add-ons? "
            "Answer with just the number.",
            check_tier_math,
        ),
        "deep-reason": (
            # Discriminates genuine reasoning from shallow pattern-matching, so a
            # fast lightweight model can't clean-sweep the heavy tier on the easy
            # probes alone (quality gate for auto-allocation).
            "If 5 machines take 5 minutes to make 5 widgets, how many minutes do "
            "100 machines take to make 100 widgets? Answer with just the number.",
            check_deep_reason,
        ),
    }


class BenchCallError(Exception):
    """A probe call that produced no scoreable answer for a harness/transport
    reason (unreachable, rate limited, auth, token budget) — recorded as an
    inconclusive `error` row, never as a model FAIL."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def anthropic_client_kwargs() -> dict:
    """Extra Anthropic client kwargs from the environment. An API key that is
    not scoped to a workspace must send `anthropic-workspace-id` on every
    request (the 2026-09-26 bench got a 400 on every baseline probe without
    it); ANTHROPIC_WORKSPACE_ID is optional and only sent when set. The value
    is an id, not a secret, but it is still never printed."""
    workspace = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
    return {"default_headers": {"anthropic-workspace-id": workspace}} if workspace else {}


def generate_anthropic(model: str, prompt: str) -> tuple[str, float, int]:
    """Claude baseline call on the identical probe. Requires the anthropic
    package + ANTHROPIC_API_KEY; callers skip baselines when unavailable."""
    import anthropic
    client = anthropic.Anthropic(**anthropic_client_kwargs())
    start = time.time()
    response = client.messages.create(
        model=model, max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start
    text = "\n".join(b.text for b in response.content if getattr(b, "text", "")).strip()
    return text, latency, int(getattr(response.usage, "output_tokens", 0))


_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Only the final answer is ever scored. Drops inline `<think>...</think>`
    blocks and, when a model's template opened the block in the prompt so the
    reply carries only the closing tag, everything up to the last `</think>`.
    A reply with no reasoning markers is returned unchanged (stripped)."""
    text = _THINK_BLOCK.sub("", text or "")
    closers = list(_THINK_CLOSE.finditer(text))
    if closers:
        text = text[closers[-1].end():]
    return text.strip()


_THINK_CACHE: dict = {}


def ollama_think_setting(base: str, api_key: str, model: str):
    """The `think` value to send for one Ollama model, from its own /api/show
    metadata (cached per base+model):

      - `thinking.values` lists `false` (glm-5.2, kimi-k3, gemma4, ...), or the
        model is not a thinker, or the metadata is missing/unreadable
        -> False: thinking off, exactly the pre-2026-09-26 behaviour;
      - `thinking.values` has NO `false` (glm-5.3, glm-5.3-flash, gpt-oss:
        levels only) -> the lowest level ("low" when offered). Such a model
        ignores think:false and, observed 2026-09-26 on glm-5.3, pours its
        reasoning trace into `response`; asking for a level routes the trace
        into the separate `thinking` field so only the answer is scored.
    Generic: no model name is special-cased here."""
    cache_key = (base, model)
    if cache_key in _THINK_CACHE:
        return _THINK_CACHE[cache_key]
    setting = False
    try:
        body, _, _ = _post_json_with_retry(base + "/api/show", _ollama_headers(api_key),
                                           {"model": model}, attempts=1)
        values = (body.get("thinking") or {}).get("values")
        if isinstance(values, list) and values and False not in values:
            levels = [v for v in values if isinstance(v, str)]
            if levels:
                setting = "low" if "low" in levels else levels[0]
    except Exception:  # noqa: BLE001 - metadata is best-effort; default to think:false
        setting = False
    _THINK_CACHE[cache_key] = setting
    return setting


def _ollama_headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def generate(base: str, api_key: str, model: str, prompt: str,
             meta: dict | None = None) -> tuple[str, float, int]:
    """One Ollama /api/generate call. `meta` (optional, filled in place)
    records the think mode used and whether the reply hit the token cap."""
    think = ollama_think_setting(base, api_key, model)
    budget = MAX_TOKENS + (THINK_HEADROOM if think else 0)
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        # think:false — thinking models otherwise burn the whole num_predict
        # budget on hidden reasoning and return an empty `response` (observed
        # live with glm-5.2 and qwen3.5 on 2026-07-17). A model that cannot
        # switch thinking off gets its lowest level instead (see
        # ollama_think_setting); its `thinking` field is never scored.
        "think": think,
        "options": {"num_predict": budget},
    }
    body, latency, attempts = _post_json_with_retry(base + "/api/generate", _ollama_headers(api_key), payload)
    truncated = body.get("done_reason") == "length"
    text = strip_reasoning(body.get("response") or "")
    if meta is not None:
        meta.update({"think": think, "truncated": truncated, "attempts": attempts})
    if not text and truncated:
        raise BenchCallError("budget", f"no answer within {budget} tokens (reasoning used the budget)")
    return text, latency, int(body.get("eval_count") or 0)


# Hosted providers benched through the OpenAI-compatible adapter below. Every
# key, base URL override and model list comes from env/CLI only — no model id
# is ever guessed here. Qwen has NO default base on purpose: its endpoint is
# account/workspace-specific and must never be published, so it comes only
# from the QWEN_BASE_URL secret — unset means the Qwen rows are skipped with a
# note, never sent to a generic endpoint. The other three defaults are the
# providers' public endpoints, plausible starting points that have NOT been
# confirmed against the account in use, so they carry an explicit [CONFIRM] —
# a wrong path/domain fails loudly (that provider's rows error out) rather
# than silently, but it should still be checked before the provider is
# offered to a client. No base URL is ever printed or written to a report.
# Google's documented OpenAI-compatibility endpoint for the Gemini API.
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"  # [CONFIRM]
# Zhipu's mainland OpenAI-compatible v4 endpoint. The international Z.ai
# brand may instead need something like https://api.z.ai/api/paas/v4 for this
# account's key — confirm which domain the account is actually provisioned
# on before relying on this default.
GLM_DEFAULT_BASE = "https://open.bigmodel.cn/api/paas/v4"  # [CONFIRM]
# xAI's OpenAI-compatible endpoint.
XAI_DEFAULT_BASE = "https://api.x.ai/v1"  # [CONFIRM]


@dataclass(frozen=True)
class CompatProvider:
    """One hosted, OpenAI-compatible bench provider: the env/CLI names that
    control it, and the label used in results and the printed report. Adding
    a new provider is one entry here — `main()` and the report loop below are
    written against this list, not against any one provider's name."""

    key: str            # internal id stored in results["models"][model]["provider"]
    label: str          # shown in the markdown report, e.g. "Qwen API"
    key_env: str        # required API key env var
    base_env: str       # optional base-URL override env var
    default_base: str   # fallback base URL ("" = none: base_env is required)
    models_env: str     # env var holding a comma list of model ids to bench
    flag: str           # CLI flag prefix -> --<flag>-models / --<flag>-list
    # How this provider is asked to minimise thinking on chat/completions.
    # Default is Qwen/DashScope's `enable_thinking: false` (the original
    # behaviour). A provider that rejects the field with HTTP 400 is retried
    # once without it.
    thinking_params: tuple = (("enable_thinking", False),)
    # Extra max_tokens for a provider whose thinking cannot be switched off and
    # whose max_tokens includes the thought tokens (see THINK_HEADROOM).
    thinking_headroom: int = 0
    # True: no fallback — the provider is skipped unless base_env is set.
    base_required: bool = False


COMPAT_PROVIDERS: list[CompatProvider] = [
    CompatProvider("qwen", "Qwen API", "QWEN_API_KEY", "QWEN_BASE_URL",
                    "", "QWEN_BENCH_MODELS", "qwen", base_required=True),
    # Gemini: Google's OpenAI-compat docs map `reasoning_effort` onto
    # thinking_level (3.x) / thinking_budget (2.5), state reasoning cannot be
    # turned off for 3.x models ("none" exists for 2.5 non-Pro only), and
    # count thought tokens inside max_output_tokens. "low" is the lowest level
    # documented for every current Gemini model, so send that and add the
    # thinking headroom — the 2026-09-26 run's 9-11-token replies were the
    # default thinking level eating the whole 300-token budget.
    CompatProvider("gemini", "Gemini API", "GEMINI_API_KEY", "GEMINI_BASE_URL",
                    GEMINI_DEFAULT_BASE, "GEMINI_BENCH_MODELS", "gemini",
                    thinking_params=(("reasoning_effort", "low"),),
                    thinking_headroom=THINK_HEADROOM),
    CompatProvider("glm", "GLM API (Zhipu/Z.ai)", "GLM_API_KEY", "GLM_BASE_URL",
                    GLM_DEFAULT_BASE, "GLM_BENCH_MODELS", "glm"),
    CompatProvider("grok", "Grok API (xAI)", "XAI_API_KEY", "XAI_BASE_URL",
                    XAI_DEFAULT_BASE, "XAI_BENCH_MODELS", "grok"),
]


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _retry_after_s(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait before the next attempt: the server's Retry-After
    (delta-seconds or HTTP-date) when present, else exponential backoff with a
    little jitter — capped at RETRY_AFTER_CAP_S either way."""
    header = (exc.headers.get("Retry-After") if exc.headers else None) or ""
    header = header.strip()
    wait = None
    if header:
        try:
            wait = float(header)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(header)
                wait = (when - dt.datetime.now(when.tzinfo)).total_seconds()
            except (TypeError, ValueError):
                wait = None
    if wait is None:
        wait = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
    return max(0.0, min(wait, RETRY_AFTER_CAP_S))


def _post_json_with_retry(url: str, headers: dict, payload: dict,
                          attempts: int = MAX_ATTEMPTS) -> tuple[dict, float, int]:
    """POST with a bounded retry on transient statuses (RETRY_STATUSES): at
    most `attempts` calls, honouring Retry-After. Returns (body, latency_s of
    the successful call only — waits are not charged to the model, attempts
    used). Any other error, or the last transient one, propagates."""
    for attempt in range(1, attempts + 1):
        start = time.time()
        try:
            body = _post_json(url, headers, payload)
            return body, time.time() - start, attempt
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUSES or attempt == attempts:
                raise
            _sleep(_retry_after_s(exc, attempt))
    raise AssertionError("unreachable")  # pragma: no cover


def generate_openai_compat(base: str, api_key: str, model: str, prompt: str,
                           thinking_params: dict | None = None, max_tokens: int | None = None,
                           meta: dict | None = None) -> tuple[str, float, int]:
    """One chat-completions call on an OpenAI-compatible API (Qwen, Gemini,
    GLM/Zhipu, Grok — any provider in COMPAT_PROVIDERS). Thinking is switched
    off (or minimised) with the provider's own documented parameter — the
    default is Qwen's `enable_thinking: false`; a provider that rejects it
    (HTTP 400) is retried once without it. Transient 429/503-class errors get
    the bounded retry in _post_json_with_retry. Returns (text,
    latency_s, completion_tokens) — the same per-call latency/token numbers
    every provider's bench row records. Only the final message content is
    scored; any inline reasoning block is stripped."""
    if thinking_params is None:
        thinking_params = {"enable_thinking": False}
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + api_key}
    payload = {"model": model, "max_tokens": max_tokens or MAX_TOKENS,
               "messages": [{"role": "user", "content": prompt}]}
    payload.update(thinking_params)
    url = base + "/chat/completions"
    try:
        body, latency, attempts = _post_json_with_retry(url, headers, payload)
    except urllib.error.HTTPError as exc:
        if exc.code != 400 or not thinking_params:
            raise
        for key in thinking_params:
            payload.pop(key, None)
        body, latency, attempts = _post_json_with_retry(url, headers, payload)
    choices = body.get("choices") or [{}]
    text = strip_reasoning((choices[0].get("message") or {}).get("content") or "")
    truncated = choices[0].get("finish_reason") == "length"
    if meta is not None:
        meta.update({"truncated": truncated, "attempts": attempts})
    if not text and truncated:
        raise BenchCallError("budget", f"no answer within {payload['max_tokens']} tokens "
                                       "(reasoning used the budget)")
    return text, latency, int((body.get("usage") or {}).get("completion_tokens") or 0)


def classify_error(exc: Exception, workspace_set: bool | None = None) -> tuple[str, str]:
    """(kind, message) for a probe call that raised. Every kind is recorded
    as an inconclusive `error` row — "could not reach / could not ask" — and
    never as a model FAIL. Kinds: auth (incl. the workspace-scope 400),
    rate_limited (429), unavailable (5xx), gone (404/410: model retired or
    renamed), timeout, network, budget, http_<code>, error."""
    if isinstance(exc, BenchCallError):
        return exc.kind, str(exc)
    text = str(exc)
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if WORKSPACE_ERROR_MARKER in text:
        if workspace_set is None:
            workspace_set = bool(os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip())
        if workspace_set:
            return "auth", ("key not scoped to a workspace and the anthropic-workspace-id "
                            "header sent from ANTHROPIC_WORKSPACE_ID was not accepted; "
                            "check the id or use a workspace-scoped key")
        return "auth", WORKSPACE_HINT
    if isinstance(status, int):
        if status in (401, 403):
            return "auth", f"HTTP {status}: authentication/permission refused ({text[:120]})"
        if status == 429:
            return "rate_limited", f"HTTP 429: rate limited after {MAX_ATTEMPTS} attempts"
        if status in (404, 410):
            return "gone", f"HTTP {status}: model not served (retired or renamed?)"
        if status >= 500:
            return "unavailable", f"HTTP {status}: provider unavailable after {MAX_ATTEMPTS} attempts"
        return f"http_{status}", text[:200]
    if isinstance(exc, (TimeoutError,)) or "timed out" in text.lower():
        return "timeout", text[:200]
    if isinstance(exc, (urllib.error.URLError, ConnectionError, OSError)):
        return "network", text[:200]
    return "error", text[:200]


# An error of one of these kinds will repeat identically on every remaining
# probe for that model, so the rest are recorded with the same reason instead
# of spending more calls on it.
FATAL_ERROR_KINDS = {"auth", "gone"}


def discover_openai_compat(base: str, api_key: str, label: str = "provider") -> list:
    """Model ids an OpenAI-compatible API lists at /models. Best-effort — this
    is the ONLY source of model ids for a provider's `--<flag>-list`, so a
    provider with a real /models endpoint never needs a guessed id in code."""
    try:
        req = urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + api_key})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8"))
        return [m.get("id") for m in body.get("data", []) if m.get("id")]
    except Exception as exc:  # noqa: BLE001 - discovery is optional, never fatal
        print(f"{label} discover: /models failed ({safe_error_text(exc)})", flush=True)
        return []


def safe_error_text(exc: Exception) -> str:
    """An exception summary that can never carry a URL: the HTTP status for
    an HTTPError, else the exception class name. (Some urllib errors embed
    the request URL, and a provider's base URL must never be published.)"""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return f"HTTP {code}"
    return type(exc).__name__


def redact(text: str, private: list) -> str:
    """Replace every private string (full base URLs and their hosts) in text."""
    import urllib.parse
    for value in private:
        if not value:
            continue
        host = urllib.parse.urlsplit(value).netloc
        for needle in (value, host):
            if needle:
                text = text.replace(needle, "<redacted>")
    return text


def discover_models(base: str, api_key: str) -> list:
    """Model names the bridge advertises via /api/tags. Lets the weekly bench
    auto-include a newly-released cloud model (e.g. Kimi K3 Max) with no code
    change and no tag guessing. Best-effort: returns [] on any error."""
    import urllib.request
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    try:
        req = urllib.request.Request(base + "/api/tags", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8"))
        return [m.get("name") for m in body.get("models", []) if m.get("name")]
    except Exception as exc:  # noqa: BLE001 - discovery is optional, never fatal
        print(f"discover: /api/tags failed ({exc}); using the static model list", flush=True)
        return []


def error_row(kind: str, message: str) -> dict:
    """An inconclusive probe: the model was not reached or not asked, so there
    is no answer to score. `pass` is None — neither a pass nor a FAIL — so a
    clean-sweep check (all probes pass) can never count it as capability, and
    the scoreboard keeps "could not reach" apart from "answered wrong"."""
    return {"pass": None, "status": "error", "error_kind": kind, "error": message[:200]}


def is_error(probe: dict) -> bool:
    return probe.get("status") == "error" or ("error" in probe and "latency_s" not in probe)


def cell_status(probe: dict) -> str:
    if is_error(probe):
        return "ERR"
    return "PASS" if probe.get("pass") else "FAIL"


def row_counts(row: dict) -> tuple[int, int, int]:
    """(passes, fails, errors) over a row's probes."""
    passes = fails = errors = 0
    for name in probes():
        r = row.get(name)
        if not isinstance(r, dict):
            continue
        if is_error(r):
            errors += 1
        elif r.get("pass"):
            passes += 1
        else:
            fails += 1
    return passes, fails, errors


def row_verdict(row: dict) -> str:
    """clean sweep | fail | inconclusive. A row with any FAIL is `fail` (a
    real wrong answer is evidence even beside an unreachable probe); a row with
    no FAIL but at least one error is `inconclusive` — never a pass, never a
    permanent disqualification (the next run decides)."""
    passes, fails, errors = row_counts(row)
    if fails:
        return "fail"
    if errors or passes < len(probes()):
        return "inconclusive"
    return "clean sweep"


def merge_rows(prior: dict, fresh: dict, same_harness: bool = True) -> dict:
    """Same-day rerun merge: fresh rows win per model, except that a fresh
    inconclusive (error) probe never overwrites a conclusive result for the
    same model and probe from earlier the same day — an outage on the rerun
    must not erase evidence already gathered today. That carry-over only
    applies when the earlier run used the same harness (same_harness): a
    score produced by an older, since-fixed call path is not evidence."""
    merged = dict(prior)
    for model, row in fresh.items():
        old = prior.get(model)
        if not isinstance(old, dict) or not same_harness:
            merged[model] = row
            continue
        combined = dict(row)
        for name in probes():
            new_probe, old_probe = row.get(name), old.get(name)
            if (isinstance(new_probe, dict) and is_error(new_probe)
                    and isinstance(old_probe, dict) and not is_error(old_probe)):
                combined[name] = old_probe
        merged[model] = combined
    return merged


def render_markdown(results: dict) -> str:
    provider_label = {provider.key: provider.label for provider in COMPAT_PROVIDERS}
    names = list(probes())
    lines = [f"# Model bench — {results['date']}", "",
             f"Base: `{results['base']}` · max_tokens={MAX_TOKENS} "
             f"(+{THINK_HEADROOM} headroom only where thinking cannot be switched off)", "",
             "PASS/FAIL = the model answered and was scored. **ERR = could not reach or "
             "could not ask** (rate limit, outage, auth, retired model, token budget) — "
             "inconclusive: never counted as a pass, never as a FAIL.", "",
             # avg latency + total tokens sit next to the probe scores so a
             # provider's speed and volume can be weighed against its pass
             # rate — capability, not price (no $/Mtok table lives here; see README).
             "| model | " + " | ".join(names) + " | score | verdict | avg latency | total tokens |",
             "|---|" + "---|" * (len(names) + 4)]
    notes: list = []
    for model, row in results["models"].items():
        cells, lats, toks = [], [], []
        errors_seen: dict = {}
        for name in names:
            r = row.get(name)
            if r is None:  # merged older row from before a probe existed
                cells.append("—")
                continue
            if is_error(r):
                kind = r.get("error_kind") or "error"
                cells.append(f"ERR ({kind})")
                errors_seen.setdefault(r.get("error", kind), []).append(name)
                continue
            cells.append(("PASS" if r.get("pass") else "FAIL") + f" {r['latency_s']}s"
                         + (" (trunc)" if r.get("truncated") else ""))
            lats.append(r["latency_s"])
            if "tokens" in r:
                toks.append(r["tokens"])
        passes, fails, errors = row_counts(row)
        score = f"{passes}/{len(names)}" + (f" · {errors} err" if errors else "")
        avg = f"{sum(lats) / len(lats):.1f}s" if lats else "-"
        total_tokens = str(sum(toks)) if toks else "-"
        label = f"**{model}** (baseline)" if row.get("baseline") else model
        if row.get("provider") in provider_label:
            label = f"{model} ({provider_label[row['provider']]})"
        lines.append(f"| {label} | " + " | ".join(cells)
                     + f" | {score} | {row_verdict(row)} | {avg} | {total_tokens} |")
        for message, probe_names in errors_seen.items():
            notes.append(f"- `{model}` ERR on {', '.join(probe_names)}: {message}")
        if row.get("think"):
            notes.append(f"- `{model}` cannot switch thinking off; benched at think=`{row['think']}`, "
                         "answer scored without the reasoning trace (not allocatable to the "
                         "bridge tier, which calls think:false).")
    if notes:
        lines += ["", "## Notes", ""] + notes
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="router v5.1 weekly model bench")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES),
                        help="Claude models benched head-to-head on the same probes "
                             "(skipped unless ANTHROPIC_API_KEY is set); '' disables")
    parser.add_argument("--base", default="")
    parser.add_argument("--out-dir", default=str(HERE / "reports"))
    parser.add_argument("--discover", action="store_true",
                        help="also bench every model the bridge lists in /api/tags "
                             "(auto-picks up NEW cloud models like Kimi K3 Max the day "
                             "they land — no tag guessing, runs PC-off in CI)")
    for provider in COMPAT_PROVIDERS:
        models_help = (f"comma list of {provider.label} model ids to bench via the "
                        f"OpenAI-compatible adapter (needs {provider.key_env}); empty skips it")
        if provider.key == "gemini":
            # Elie's first Gemini candidate (2026-09-26) is "Gemini 3.8
            # Flash", but its exact API model id is UNCONFIRMED — get it from
            # --gemini-list once GEMINI_API_KEY exists; never guess it here.
            models_help += " ([CONFIRM: exact Gemini 3.8 Flash model ID] via --gemini-list)"
        parser.add_argument(f"--{provider.flag}-models", default=os.getenv(provider.models_env, ""),
                            help=models_help)
        parser.add_argument(
            f"--{provider.flag}-list", action="store_true",
            help=f"print the model ids the {provider.label} /models endpoint lists, then exit",
        )
    args = parser.parse_args()

    # Resolve each hosted provider's key/base once, and its bench model list
    # (env or --<flag>-models) — a provider without a key is skipped cleanly
    # (a printed reason, never an exception that kills the whole bench). Keys
    # are read into local variables only and never printed.
    compat_key: dict[str, str] = {}
    compat_base: dict[str, str] = {}
    model_provider: dict[str, str] = {}  # model id -> provider key, for dispatch below
    compat_models_by_provider: dict[str, list] = {}
    for provider in COMPAT_PROVIDERS:
        key = os.getenv(provider.key_env, "").strip()
        base_url = (os.getenv(provider.base_env, "").strip() or provider.default_base).rstrip("/")
        compat_key[provider.key] = key
        compat_base[provider.key] = base_url
        base_missing = provider.base_required and not base_url
        base_note = (f"{provider.base_env} unset — {provider.label} skipped (its endpoint "
                     "comes only from that secret; no fallback to a generic endpoint)")

        if getattr(args, f"{provider.flag}_list"):
            if not key:
                print(f"{provider.key_env} unset", flush=True)
                return 1
            if base_missing:
                print(base_note, flush=True)
                return 1
            print("\n".join(discover_openai_compat(base_url, key, label=provider.key)))
            return 0

        provider_models = [m.strip() for m in getattr(args, f"{provider.flag}_models").split(",") if m.strip()]
        if provider_models and not key:
            print(f"note: {provider.key_env} unset — {provider.label} models skipped", flush=True)
            provider_models = []
        elif provider_models and base_missing:
            print(f"note: {base_note}", flush=True)
            provider_models = []
        compat_models_by_provider[provider.key] = provider_models
        for model in provider_models:
            model_provider[model] = provider.key

    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    base = (args.base or ("https://ollama.com" if api_key else "http://localhost:11434")).rstrip("/")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.discover:
        found = discover_models(base, api_key)
        added = [m for m in found if m not in models]
        if added:
            print(f"discover: +{len(added)} new model(s): {', '.join(added)}", flush=True)
        models = list(dict.fromkeys(models + found))
    baselines = [m.strip() for m in args.baselines.split(",") if m.strip()]
    if baselines and not os.getenv("ANTHROPIC_API_KEY", "").strip():
        print("note: ANTHROPIC_API_KEY unset — Claude baselines skipped", flush=True)
        baselines = []
    today = dt.date.today().isoformat()
    all_compat_models = [m for provider in COMPAT_PROVIDERS for m in compat_models_by_provider[provider.key]]

    if baselines and not os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip():
        print("note: ANTHROPIC_WORKSPACE_ID unset — fine for a workspace-scoped key; "
              "a key that is not workspace-scoped will show as an auth error row", flush=True)
    provider_by_key = {provider.key: provider for provider in COMPAT_PROVIDERS}

    results: dict = {"date": today, "base": base, "harness": HARNESS_VERSION, "models": {}}
    for model in models + baselines + all_compat_models:
        is_baseline = model in baselines
        provider_key = None if is_baseline else model_provider.get(model)
        row: dict = {"baseline": is_baseline} if is_baseline else {}
        if provider_key:
            row["provider"] = provider_key
        fatal: tuple | None = None
        for name, (prompt, check) in probes().items():
            meta: dict = {}
            if fatal:
                # Same reason would repeat on every remaining probe (auth,
                # retired model) — record it, don't spend more calls.
                row[name] = error_row(*fatal)
            else:
                try:
                    if is_baseline:
                        text, latency, tokens = generate_anthropic(model, prompt)
                    elif provider_key:
                        provider = provider_by_key[provider_key]
                        text, latency, tokens = generate_openai_compat(
                            compat_base[provider_key], compat_key[provider_key], model, prompt,
                            thinking_params=dict(provider.thinking_params),
                            max_tokens=MAX_TOKENS + provider.thinking_headroom, meta=meta)
                    else:
                        text, latency, tokens = generate(base, api_key, model, prompt, meta=meta)
                    # latency_s and tokens are recorded for every model on every
                    # probe (all providers, including the ones added here) so
                    # capability, speed and volume can be weighed together once a
                    # provider's $/Mtok pricing is confirmed — see the README note
                    # on why no price table lives in this script.
                    passed = bool(check(text))
                    row[name] = {"pass": passed, "status": "pass" if passed else "fail",
                                 "latency_s": round(latency, 1), "tokens": tokens,
                                 "reply_head": text[:120]}
                    if meta.get("truncated"):
                        row[name]["truncated"] = True
                    if meta.get("attempts", 1) > 1:
                        row[name]["attempts"] = meta["attempts"]
                except Exception as exc:  # noqa: BLE001 - a dead model must not kill the bench
                    kind, message = classify_error(exc)
                    # Error text goes into a committed (public) report: strip
                    # any provider base URL / host that an exception carried.
                    message = redact(message, list(compat_base.values()))
                    row[name] = error_row(kind, message)
                    if kind in FATAL_ERROR_KINDS:
                        fatal = (kind, message)
            if meta.get("think"):
                row["think"] = meta["think"]
            print(f"{model:24s} {name:14s} {cell_status(row[name]):4s} "
                  f"{row[name].get('latency_s', row[name].get('error_kind', '-'))}", flush=True)
        results["models"][model] = row

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Same-day reruns MERGE per-model (fresh rows win) instead of clobbering
    # the whole report — a partial re-test must not erase the full table.
    json_path = out_dir / f"{today}.json"
    if json_path.exists():
        try:
            prior = json.loads(json_path.read_text(encoding="utf-8"))
            results["models"] = merge_rows(prior.get("models", {}), results["models"],
                                           same_harness=prior.get("harness") == HARNESS_VERSION)
        except (ValueError, OSError):
            pass  # unreadable prior report: overwrite it
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / f"{today}.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"\nReport: {out_dir / (today + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
