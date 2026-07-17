"""Claude model router v5.1 — multi-engine registry + dispatch + usage log.

This router keeps the public/free kit simple while tracking the current
lineup across two engines (Anthropic API + the Ollama bridge):
- Claude Haiku 4.5 for mechanical/high-volume work.
- Claude Sonnet 5 as the default workhorse.
- GLM 5.2 (Ollama bridge) between Sonnet and Opus for heavy NON-stakes bulk
  reasoning/drafting — protects Claude quota. NUMBERS RULE: never routes
  customer-facing price/quote/invoice/legal work; pass stakes=True to keep
  a task on Claude tiers entirely.
- Claude Opus 4.8 for complex agentic coding and enterprise-quality work.
- Claude Fable 5 as the frontier reserve tier.

Engines and how they resolve (the v5.1 setup contract):
- Anthropic engine: needs `pip install anthropic` + ANTHROPIC_API_KEY (or
  ANTHROPIC_AUTH_TOKEN). Without them the router still works in OFFLINE
  mode — everything routes to the Ollama bridge, Claude tiers are skipped
  in escalation, and stakes=True refuses (stakes never runs on the bridge).
- Ollama bridge: tried as a chain, first reachable base wins —
    1. CLAUDE_ROUTER_OLLAMA_URL if set (your local daemon OR a routing
       server on your tailnet, e.g. http://<tailscale-host>:11434),
       otherwise http://localhost:11434;
    2. https://ollama.com when OLLAMA_API_KEY is set (Ollama Cloud direct).
  A daemon signed in to an Ollama account runs `:cloud` tags through the
  local endpoint, so pointing the URL at that daemon covers both local and
  cloud models with no API key on the client.

Usage:
    from router import run
    reply = run("Refactor this function", tier="sonnet", effort="high")

CLI:
    python router.py --doctor           # setup check: engines, bases, tags
    python router.py --registry         # active registry as JSON
    python router.py "task..." [--tier glm] [--stakes] [--effort high]

Logs every dispatch to router-usage.csv (CLAUDE_ROUTER_LOG overrides).
"""

from __future__ import annotations

import csv
import datetime as dt
import ipaddress
import json
import os
import pathlib
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Optional

try:  # the Anthropic engine is optional in OFFLINE (Ollama-only) mode
    import anthropic
except ImportError:  # pragma: no cover - exercised on machines without the SDK
    anthropic = None  # type: ignore[assignment]

_CLIENT = None

# Short timeout for "is this base up" probes; long timeout for generation
# (cloud-sized models can legitimately take minutes on big prompts).
PROBE_TIMEOUT = 4
GENERATE_TIMEOUT = int(os.getenv("CLAUDE_ROUTER_OLLAMA_TIMEOUT", "300") or "300")


class RouterSetupError(RuntimeError):
    """The router cannot run this task with the current environment.

    The message always says what to set/start. `python router.py --doctor`
    prints the full setup table.
    """


# --------------------------------------------------------------------------- #
# NUMBERS RULE — stakes keyword backstop (end-to-end, not caller-optional)
# --------------------------------------------------------------------------- #
# CUSTOMER money/legal wording forces Claude-only even when the caller forgot
# stakes=True. HIGH-PRECISION and lenient by design: a code review, a model
# valuation, a top-to-bottom audit, or refactoring the pricing MODULE is NOT
# stakes and runs happily on the week's strongest bridge model. Only genuine
# customer commerce/legal + an actual dollar amount is caught. Keep in step with
# hq_orchestrator/core.py's _STAKES_HINT_RE (intentional duplication — the two
# live in different packages and must not import across).
_STAKES_HINT_RE = re.compile(
    r"(?i)"
    r"\b(invoice|refund|chargeback|remittance|payable|payslip|superannuation)\b"
    r"|\b(tax\s+invoice|purchase\s+order|payment\s+link|credit\s+card|bank\s+details|bsb|abn|gst)\b"
    r"|\bliability\b|\bindemnif|terms\s+(?:and|&)\s+conditions|\blegal\s+(?:advice|letter|contract)\b"
    r"|\$\s?\d"
)


def looks_like_stakes(task: str) -> bool:
    """True when task text carries customer money/legal signal (see the regex).
    Extra terms can be added per-deployment via CLAUDE_ROUTER_STAKES_KEYWORDS
    (comma-separated, matched case-insensitively as whole words)."""
    if not isinstance(task, str):
        return False
    if _STAKES_HINT_RE.search(task):
        return True
    extra = os.getenv("CLAUDE_ROUTER_STAKES_KEYWORDS", "").strip()
    if extra:
        terms = [re.escape(t.strip()) for t in extra.split(",") if t.strip()]
        if terms and re.search(r"(?i)\b(" + "|".join(terms) + r")\b", task):
            return True
    return False


# --------------------------------------------------------------------------- #
# SSRF guard — the Ollama base URL is attacker-influencable (env), so validate
# it before urllib ever touches it. Blocks file://, cloud-metadata IPs, and
# arbitrary public hosts that would exfiltrate the prompt + Ollama key.
# --------------------------------------------------------------------------- #
_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_HOST_SUFFIXES = (".ts.net",)      # Tailscale MagicDNS
_ALLOWED_HOSTNAMES = {"localhost", "ollama.com"}


def _extra_allowed_hosts() -> set[str]:
    # Strip an optional :port so an allowlist entry like "host:11434" matches
    # parsed.hostname (which has no port) — Kimi review #8.
    hosts: set[str] = set()
    for h in os.getenv("CLAUDE_ROUTER_OLLAMA_ALLOW_HOSTS", "").split(","):
        h = h.strip().lower()
        if not h:
            continue
        if h.count(":") == 1:  # host:port (not a bare IPv6 literal)
            h = h.rsplit(":", 1)[0]
        hosts.add(h)
    return hosts


def _host_resolves_private(host: str) -> bool:
    """True iff EVERY DNS resolution of host is loopback / RFC1918 / Tailscale
    CGNAT. The authoritative anti-SSRF check at DISPATCH time — it closes both
    the single-label-name gap (a bare name resolving to a public host) and DNS
    rebinding (an allowed name flipping to 169.254.169.254 between check and
    use). Kimi review #1 + #5. Fail closed on resolution error."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    saw = False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])  # strip IPv6 zone id
        except ValueError:
            return False
        saw = True
        if ip.is_link_local:
            return False  # 169.254.169.254 metadata etc. — is_private also covers
            #               link-local, so this MUST be checked first.
        if ip.is_loopback or ip.is_private or ip in ipaddress.ip_network("100.64.0.0/10"):
            continue
        return False  # a public IP among the results -> reject
    return saw


def is_allowed_base(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Allow: http/https to loopback, RFC1918/ULA private,
    Tailscale CGNAT (100.64/10) or *.ts.net, ollama.com, or an explicit host in
    CLAUDE_ROUTER_OLLAMA_ALLOW_HOSTS. Reject everything else — notably
    file://, 169.254.169.254 (cloud metadata), and arbitrary public URLs."""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:
        return False, f"unparseable URL: {exc}"
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"scheme '{parsed.scheme}' not allowed (http/https only)"
    host = parsed.hostname
    if not host:
        return False, "no host in URL"
    host_l = host.lower()
    if host_l in _extra_allowed_hosts():
        return True, "explicitly allowlisted host"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_loopback:
            return True, "loopback"
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return True, "Tailscale CGNAT range"
        if ip.is_link_local:                       # 169.254/16 incl. cloud metadata
            return False, "link-local/metadata IP blocked (SSRF)"
        if ip.is_private:                          # 10/8, 172.16/12, 192.168/16, fc00::/7
            return True, "private network"
        return False, f"public IP {host} blocked (SSRF) — allowlist it explicitly if intended"
    if host_l in _ALLOWED_HOSTNAMES or host_l.endswith(_ALLOWED_HOST_SUFFIXES):
        return True, "allowed hostname"
    if "." not in host_l:
        # A single-label hostname (no dot) is a LAN/local name — it cannot be a
        # public internet domain, so it can't exfiltrate to the open internet.
        return True, "single-label LAN hostname"
    return False, f"public host '{host}' blocked (SSRF) — set CLAUDE_ROUTER_OLLAMA_ALLOW_HOSTS to permit it"


def _log_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent / os.getenv("CLAUDE_ROUTER_LOG", "router-usage.csv")


@dataclass(frozen=True)
class ModelInfo:
    api_id: str
    label: str
    role: str
    context_window: str
    max_output_tokens: int
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    supports_effort: bool
    availability: str = "generally_available"
    engine: str = "anthropic"  # "anthropic" | "ollama"


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "haiku": ModelInfo(
        api_id="claude-haiku-4-5-20251001",
        label="Claude Haiku 4.5",
        role="Mechanical work, fast sub-agents, extraction, formatting, high-volume low-latency tasks.",
        context_window="200k",
        max_output_tokens=64_000,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=5.0,
        supports_effort=False,
    ),
    "sonnet": ModelInfo(
        api_id="claude-sonnet-5",
        label="Claude Sonnet 5",
        role="Default workhorse for coding, data analysis, drafting, agentic tool use, and daily production tasks.",
        context_window="1M",
        max_output_tokens=128_000,
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        supports_effort=True,
    ),
    "glm": ModelInfo(
        api_id="glm-5.2:cloud",
        label="GLM 5.2 (Ollama bridge)",
        role="Heavy NON-stakes bulk work between Sonnet and Opus: long drafting, summarising, research digests, bulk analysis. Never customer-facing numbers or legal.",
        context_window="200k",
        max_output_tokens=32_000,
        input_usd_per_mtok=0.0,   # billed via Ollama Cloud subscription / local, not per-token API
        output_usd_per_mtok=0.0,
        supports_effort=False,
        availability="ollama-bridge",
        engine="ollama",
    ),
    "opus": ModelInfo(
        api_id="claude-opus-4-8",
        label="Claude Opus 4.8",
        role="Complex agentic coding, architecture, enterprise-quality analysis, large refactors, high-accuracy work.",
        context_window="1M",
        max_output_tokens=128_000,
        input_usd_per_mtok=5.0,
        output_usd_per_mtok=25.0,
        supports_effort=True,
    ),
    "fable": ModelInfo(
        api_id="claude-fable-5",
        label="Claude Fable 5",
        role="Frontier reserve for the hardest reasoning, long-running agents, novel system design, and failed Opus cases.",
        context_window="1M",
        max_output_tokens=128_000,
        input_usd_per_mtok=10.0,
        output_usd_per_mtok=50.0,
        supports_effort=True,
    ),
}

LADDER = ["haiku", "sonnet", "glm", "opus", "fable"]

ESCALATE = {
    "haiku": "sonnet",
    "sonnet": "glm",   # quota-saving middle step; skipped when stakes=True
    "glm": "opus",
    "opus": "fable",
    "fable": "fable",
}

FALLBACK = {
    "fable": "opus",
    "opus": "sonnet",
    "glm": "sonnet",   # bridge unreachable -> back to Claude, never block
    "sonnet": "haiku",
    "haiku": "haiku",
}

# Signals that a reply is not a complete, confident answer. Policy hard rule:
# one failed or incomplete response at a tier => escalate immediately.
REFUSAL_PREFIXES = ("i can't", "i cannot", "i'm unable", "i am unable", "sorry, i can't", "sorry, i cannot")

SUPPORTED_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}

CLASSIFIER_PROMPT = """You are a model routing classifier.
Reply with exactly one word: HAIKU, SONNET, GLM, OPUS, or FABLE.

HAIKU  — mechanical only: reformat, rename, extract, boilerplate, row cleanup, one-line answers.
SONNET — default: coding, drafting, data analysis, business tasks, multi-step agent work, tool use.
GLM    — heavy NON-stakes bulk: long summaries, research digests, big first-draft documents, bulk rewriting. NEVER anything with customer-facing prices, quotes, invoices, or legal content.
OPUS   — quality-critical: complex architecture, deep analysis, large refactors, enterprise/customer-facing work.
FABLE  — frontier reserve: hardest reasoning, novel system design, long-running agents, or failed Opus attempts.

Task:
{task}
"""


# --------------------------------------------------------------------------- #
# Engine availability
# --------------------------------------------------------------------------- #
def anthropic_ready() -> bool:
    """True when the Anthropic engine can actually take a request: SDK
    installed and an auth env var set. Does not validate the key (no spend)."""
    if anthropic is None:
        return False
    return bool(
        os.getenv("ANTHROPIC_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
    )


def _client():
    global _CLIENT
    if anthropic is None:
        raise RouterSetupError(
            "the 'anthropic' package is not installed — run: pip install anthropic"
        )
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def _create_message(request: dict):
    """messages.create with a graceful fallback for the effort control
    (Kimi review #2): `output_config={"effort": ...}` is a newer/beta field —
    on an SDK or API that doesn't accept it, retry once WITHOUT it rather than
    hard-failing the whole dispatch. Effort is an optimisation, not a
    requirement; every other error propagates untouched."""
    try:
        return _client().messages.create(**request)
    except TypeError as exc:
        if "output_config" in request and "output_config" in str(exc):
            request.pop("output_config", None)
            return _client().messages.create(**request)
        raise
    except anthropic.APIStatusError as exc:
        if ("output_config" in request and getattr(exc, "status_code", None) == 400
                and "output_config" in str(getattr(exc, "message", "") or exc).lower()):
            request.pop("output_config", None)
            return _client().messages.create(**request)
        raise


def _ollama_bases() -> list[tuple[str, str]]:
    """Ordered (base_url, api_key) chain for the Ollama bridge.

    1. CLAUDE_ROUTER_OLLAMA_URL — one URL or a comma-separated priority list
       (e.g. a tailnet routing server, then a second PC's daemon), defaulting
       to http://localhost:11434 when unset.
    2. https://ollama.com when OLLAMA_API_KEY is set (Ollama Cloud direct),
       unless it is already listed.
    The key rides along on every base when set — a plain daemon ignores it;
    ollama.com and any self-hosted auth proxy require it.
    """
    key = os.getenv("OLLAMA_API_KEY", "").strip()
    raw = os.getenv("CLAUDE_ROUTER_OLLAMA_URL", "").strip()
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()] or ["http://localhost:11434"]
    if key and "https://ollama.com" not in urls:
        urls.append("https://ollama.com")
    seen: set[str] = set()
    bases: list[tuple[str, str]] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        ok, reason = is_allowed_base(url)
        if not ok:
            # SSRF guard: never dispatch to a rejected base. Warn loudly so a
            # misconfigured URL is visible rather than silently swallowed.
            print(f"[router] BLOCKED Ollama base {url!r}: {reason}", file=sys.stderr, flush=True)
            continue
        bases.append((url, key))
    return bases


def _ollama_configured() -> bool:
    """Cheap 'worth trying the bridge' check used for offline rerouting:
    an explicit URL or Cloud key means yes; otherwise probe localhost once."""
    if os.getenv("CLAUDE_ROUTER_OLLAMA_URL", "").strip() or os.getenv("OLLAMA_API_KEY", "").strip():
        return True
    return _probe_base("http://localhost:11434", "") is not None


def _probe_base(base: str, api_key: str) -> Optional[str]:
    """Return the Ollama version string when `base` answers, else None."""
    request = urllib.request.Request(base + "/api/version", headers=_auth_headers(api_key))
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body.get("version", "unknown"))
    except Exception:
        return None


def _auth_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


# --------------------------------------------------------------------------- #
# Visibility — always know WHERE a reply came from
# --------------------------------------------------------------------------- #
def _via_label(base: Optional[str], model_tag: str) -> str:
    """Human-readable 'where did this run' label.

    Nuance worth being honest about: a ':cloud' tag served through a LOCAL
    daemon still computes on Ollama Cloud — the daemon just proxies it. The
    label says both the endpoint and (when inferable) where compute happened.
    """
    if base is None:
        return "Anthropic API (ONLINE)"
    cloud_tag = model_tag.endswith(":cloud")
    if base == "https://ollama.com":
        return "Ollama Cloud (ONLINE)"
    host = base.split("//", 1)[-1].split(":")[0]
    local = host in ("localhost", "127.0.0.1")
    where = "local daemon" if local else f"routing server {host} (tailnet/LAN)"
    if cloud_tag:
        return f"{where} -> ':cloud' tag, compute on Ollama Cloud (ONLINE)"
    return f"{where} (OFFLINE/on-prem)"


def _announce(tier: str, model_id: str, via: str, seconds: float, out_tok: int, status: str) -> None:
    """One stderr line per dispatch so the caller always SEES the routing.
    stderr, not stdout — the answer text stays clean for piping.
    Silence with CLAUDE_ROUTER_ANNOUNCE=0."""
    if os.getenv("CLAUDE_ROUTER_ANNOUNCE", "1").strip().lower() in {"0", "false", "off"}:
        return
    print(f"[router] {tier} -> {model_id} via {via} | {seconds:.1f}s, {out_tok} tok, {status}",
          file=sys.stderr, flush=True)


# Written after every dispatch; read by the Claude Code status line
# (claude_status_line.py) so the desktop app shows online/offline live.
ROUTING_STATUS_FILE = pathlib.Path.home() / ".claude" / ".routing-status.json"


def _engine_of(via: str) -> str:
    """Collapse a via-label to the status-line engine key: anthropic | cloud | local."""
    if "Anthropic" in via:
        return "anthropic"
    if "Ollama Cloud (ONLINE)" in via or "compute on Ollama Cloud" in via:
        return "cloud"
    return "local"


def _write_routing_status(tier: str, model_id: str, via: str) -> None:
    """Record the last dispatch for the status line. Never raises — a status
    write must never break a real dispatch. Disable with CLAUDE_ROUTER_STATUS=0."""
    if os.getenv("CLAUDE_ROUTER_STATUS", "1").strip().lower() in {"0", "false", "off"}:
        return
    try:
        ROUTING_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ROUTING_STATUS_FILE.write_text(json.dumps({
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "engine": _engine_of(via),
            "model": model_id,
            "tier": tier,
            "via": via,
        }), encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def registry() -> dict[str, dict[str, object]]:
    """Return the active model registry, including environment overrides."""
    data: dict[str, dict[str, object]] = {}
    for tier, info in MODEL_REGISTRY.items():
        row = asdict(info)
        row["active_api_id"] = _model_id(tier)
        row["enabled"] = _tier_enabled(tier)
        data[tier] = row
    return data


def classify(task: str) -> str:
    """Classify a task into a router tier without running the final task."""
    return _classify(task)


def run(
    task: str,
    max_tokens: int = 4096,
    tier: Optional[str] = None,
    effort: Optional[str] = None,
    stakes: bool = False,
) -> str:
    """Run a task through the model router.

    Args:
        task: User task or prompt.
        max_tokens: Response token cap.
        tier: Manual tier override: haiku, sonnet, glm, opus, fable.
        effort: Optional effort override for supported models: low, medium, high, xhigh, max.
        stakes: True keeps the task on Claude tiers only (customer-facing
            numbers, quotes, invoices, legal — the NUMBERS RULE): the glm
            tier is skipped in classification and escalation, and OFFLINE
            mode refuses rather than downgrade a stakes task to the bridge.
            Even when False, a customer money/legal keyword in the task text
            forces the same behaviour (looks_like_stakes) — so a forgotten
            flag can never leak an invoice/refund to the open-weight bridge.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    # End-to-end NUMBERS RULE: the flag is a floor, not the only trigger. A
    # money/legal signal in the text promotes the task to stakes regardless.
    effective_stakes = bool(stakes) or looks_like_stakes(task)
    if effective_stakes and not stakes:
        print("[router] stakes keyword detected — forcing Claude-only (NUMBERS RULE)",
              file=sys.stderr, flush=True)

    claude_ok = anthropic_ready()
    current_tier = _normalise_tier(tier) if tier else _classify(task)
    current_tier = _apply_stakes(current_tier, effective_stakes)
    if not claude_ok:
        current_tier = _reroute_offline(current_tier, effective_stakes)
    tried: set[str] = set()
    last_text = ""

    for _attempt in range(len(LADDER)):
        tried.add(current_tier)
        model_id = _model_id(current_tier)
        requested_effort = _effort_for(current_tier, effort)
        stop_reason: Optional[str] = None

        started = time.time()
        via = "?"
        if MODEL_REGISTRY[current_tier].engine == "ollama":
            try:
                text, input_tokens, output_tokens, used_base, stop_reason = _ollama_generate(model_id, task, max_tokens)
                via = _via_label(used_base, model_id)
            except Exception as exc:
                # Bridge down or tag missing = infra failure, not capability:
                # recover on Sonnet if we haven't already tried it, otherwise
                # keep climbing. The tried-set prevents sonnet<->glm ping-pong.
                _log(current_tier, model_id, "", 0, 0, "fallback_bridge", via="bridge unreachable")
                _announce(current_tier, model_id, "Ollama bridge UNREACHABLE — falling back",
                          time.time() - started, 0, "fallback")
                recovery = (
                    "sonnet"
                    if claude_ok and "sonnet" not in tried
                    else _next_tier_up(current_tier, tried, effective_stakes, claude_ok)
                )
                if recovery is None:
                    if last_text:
                        return last_text
                    raise RouterSetupError(
                        f"Ollama bridge failed and no other engine is available: {exc}. "
                        "Run `python router.py --doctor` for the setup table."
                    ) from exc
                current_tier = recovery
                continue
        else:
            request: dict[str, object] = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": task}],
            }
            if requested_effort:
                request["output_config"] = {"effort": requested_effort}
            try:
                response = _create_message(request)
            except anthropic.APIStatusError as exc:
                # Fable can be unavailable or permission-restricted on some accounts.
                # Fall back one tier for access/availability errors, but re-raise all other errors.
                if getattr(exc, "status_code", None) in {400, 401, 403, 404} and current_tier == "fable":
                    _log(current_tier, model_id, requested_effort or "", 0, 0,
                         f"fallback_{exc.status_code}", via="Anthropic API")
                    current_tier = FALLBACK[current_tier]
                    continue
                raise
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            text = _response_text(response)
            stop_reason = getattr(response, "stop_reason", None)
            via = _via_label(None, model_id)

        last_text = text

        # Policy hard rule: ONE failed or incomplete response at a tier =>
        # escalate immediately (empty/short, token-cap truncation, refusal).
        if _is_incomplete(text, stop_reason):
            status = "escalated" if _next_tier_up(current_tier, tried, effective_stakes, claude_ok) else "incomplete_at_top"
            next_tier = _next_tier_up(current_tier, tried, effective_stakes, claude_ok)
            _log(current_tier, model_id, requested_effort or "", input_tokens, output_tokens,
                 status, via=via)
            _write_routing_status(current_tier, model_id, via)
            _announce(current_tier, model_id, via, time.time() - started, output_tokens, status)
            if next_tier is None:
                return text  # top of ladder: surface what we have, honestly logged
            current_tier = next_tier
            continue

        _log(current_tier, model_id, requested_effort or "", input_tokens, output_tokens, "ok", via=via)
        _write_routing_status(current_tier, model_id, via)
        _announce(current_tier, model_id, via, time.time() - started, output_tokens, "ok")
        return text

    return last_text


def doctor() -> dict:
    """Setup check for both engines — the runnable version of the by-hand
    'is this configured' table. Never spends tokens and never prints secrets.

    Returns {"rows": [...], "claude_ready": bool, "bridge_ready": bool,
             "bridge_base": str | None, "mode": str, "ok": bool}.
    """
    rows: list[dict[str, object]] = []

    sdk_ok = anthropic is not None
    rows.append({
        "check": "anthropic package",
        "ok": sdk_ok,
        "detail": "installed" if sdk_ok else "not importable",
        "fix": "" if sdk_ok else "pip install anthropic",
    })
    key_ok = bool(
        os.getenv("ANTHROPIC_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
    )
    rows.append({
        "check": "ANTHROPIC_API_KEY",
        "ok": key_ok,
        "detail": "set (validity not checked — doctor makes no API calls)" if key_ok else "unset",
        "fix": "" if key_ok else "export ANTHROPIC_API_KEY=sk-... (or ANTHROPIC_AUTH_TOKEN)",
    })
    claude_ready = sdk_ok and key_ok

    bridge_base: Optional[str] = None
    bridge_key = ""
    for base, api_key in _ollama_bases():
        version = _probe_base(base, api_key)
        reachable = version is not None
        if base == "https://ollama.com" and reachable and not api_key:
            # ollama.com answers /api/version unauthenticated, but generation
            # is rejected without a key — reachable is NOT ready here.
            rows.append({
                "check": f"Ollama @ {base}",
                "ok": False,
                "detail": f"reachable (v{version}) but OLLAMA_API_KEY is unset — "
                          "generation would be rejected",
                "fix": "export OLLAMA_API_KEY=... (create one at ollama.com -> settings -> keys)",
            })
            continue
        if base == "https://ollama.com":
            fix = "" if reachable else "check OLLAMA_API_KEY / network"
        elif base == "http://localhost:11434":
            fix = "" if reachable else (
                "start Ollama here, or set CLAUDE_ROUTER_OLLAMA_URL to your routing "
                "server (e.g. http://<tailscale-host>:11434 with 'Expose Ollama to "
                "the network' enabled), or set OLLAMA_API_KEY for Ollama Cloud"
            )
        else:
            fix = "" if reachable else (
                "check the daemon on that host is running, 'Expose Ollama to the "
                "network' is on, and this machine is on the same tailnet/VPN"
            )
        rows.append({
            "check": f"Ollama @ {base}",
            "ok": reachable,
            "detail": f"reachable (v{version})" if reachable else "unreachable",
            "fix": fix,
        })
        if reachable and bridge_base is None:
            bridge_base, bridge_key = base, api_key
    bridge_ready = bridge_base is not None

    glm_tag = _model_id("glm")
    if os.getenv("CLAUDE_ROUTER_GLM_MODEL", "").strip() or os.getenv("GLM_OLLAMA_TAG", "").strip():
        alloc_detail = f"'{glm_tag}' pinned by env override"
    else:
        allocated = bench_allocation()
        if allocated:
            alloc_detail = (f"'{glm_tag}' auto-allocated from bench {allocated['date']} "
                            f"(clean sweep, avg {allocated['avg_latency_s']}s)")
        else:
            alloc_detail = f"'{glm_tag}' registry default (no bench report / auto-allocate off)"
    rows.append({"check": "glm allocation", "ok": True, "detail": alloc_detail, "fix": ""})

    if bridge_ready:
        listed = _tag_listed(bridge_base or "", bridge_key, glm_tag)
        detail = f"'{glm_tag}' listed on {bridge_base}" if listed else (
            f"'{glm_tag}' not in /api/tags on {bridge_base} — a ':cloud' tag on a "
            "signed-in daemon may still run; otherwise: ollama pull " + glm_tag
        )
        rows.append({"check": "GLM model tag", "ok": True, "detail": detail, "fix": ""})
    else:
        rows.append({
            "check": "GLM model tag",
            "ok": False,
            "detail": f"'{glm_tag}' — cannot check, no ready Ollama base",
            "fix": "bring a bridge base up first (rows above)",
        })

    log_dir_ok = _log_path().parent.exists()
    rows.append({
        "check": "usage log",
        "ok": log_dir_ok,
        "detail": str(_log_path()),
        "fix": "" if log_dir_ok else "set CLAUDE_ROUTER_LOG to a writable path",
    })

    if claude_ready and bridge_ready:
        mode = "full ladder (Claude tiers + Ollama bridge)"
    elif claude_ready:
        mode = "claude-only (bridge offline — glm falls back to sonnet)"
    elif bridge_ready:
        mode = "OFFLINE (Ollama-only — Claude tiers skipped, stakes tasks refused)"
    else:
        mode = "unusable — no engine available"

    return {
        "rows": rows,
        "claude_ready": claude_ready,
        "bridge_ready": bridge_ready,
        "bridge_base": bridge_base,
        "mode": mode,
        "ok": claude_ready or bridge_ready,
    }


def _tag_listed(base: str, api_key: str, tag: str) -> bool:
    request = urllib.request.Request(base + "/api/tags", headers=_auth_headers(api_key))
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    names = {str(m.get("name", "")) for m in body.get("models") or []}
    return tag in names or tag.split(":")[0] in {n.split(":")[0] for n in names}


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _apply_stakes(tier: str, stakes: bool) -> str:
    """NUMBERS RULE enforcement: stakes work never lands on the Ollama bridge."""
    if stakes and MODEL_REGISTRY[tier].engine == "ollama":
        return "sonnet"
    return tier


def _reroute_offline(tier: str, stakes: bool) -> str:
    """No Anthropic engine: keep bridge tiers, reroute Claude tiers to the
    bridge for non-stakes work, and refuse stakes work outright."""
    if MODEL_REGISTRY[tier].engine == "ollama":
        return tier
    if stakes:
        raise RouterSetupError(
            "stakes=True requires a Claude tier (NUMBERS RULE: stakes never runs on "
            "the Ollama bridge), but the Anthropic engine is unavailable — install "
            "the 'anthropic' package and set ANTHROPIC_API_KEY."
        )
    if _ollama_configured():
        _log(tier, _model_id(tier), "", 0, 0, "rerouted_offline")
        return "glm"
    raise RouterSetupError(
        "no engine available: ANTHROPIC_API_KEY is unset and no Ollama bridge is "
        "configured or reachable. Run `python router.py --doctor` for the setup table."
    )


def _is_incomplete(text: str, stop_reason: Optional[str]) -> bool:
    """True when a reply should trigger the one-strike escalation: token-cap
    truncation, an outright refusal, or a suspiciously short answer that does
    NOT look deliberately terse. A short reply ending in sentence punctuation
    ("Done.", "42.", "Yes!") is treated as complete — don't burn quota
    escalating it (Kimi review #9)."""
    stripped = text.strip()
    if stop_reason == "max_tokens":
        return True
    if stripped.lower().startswith(REFUSAL_PREFIXES):
        return True
    if len(stripped) < 20 and not stripped.endswith((".", "!", "?")):
        return True
    return False


def _next_tier_up(tier: str, tried: set[str], stakes: bool, claude_ok: bool = True) -> Optional[str]:
    """Next tier strictly up the ladder that hasn't been tried this run,
    honouring the stakes guard and engine availability. None when exhausted."""
    for candidate in LADDER[LADDER.index(tier) + 1:]:
        if candidate in tried:
            continue
        if stakes and MODEL_REGISTRY[candidate].engine == "ollama":
            continue
        if not claude_ok and MODEL_REGISTRY[candidate].engine == "anthropic":
            continue
        return candidate
    return None


def _ollama_generate(model_tag: str, prompt: str, max_tokens: int) -> tuple[str, int, int, str]:
    """Dispatch to the first reachable base in the Ollama chain (routing
    server / local daemon first, then Ollama Cloud — see _ollama_bases).
    Returns (text, input_tokens, output_tokens, base_used) so callers can
    surface WHERE the reply came from."""
    bases = _ollama_bases()
    errors: list[str] = []
    for base, api_key in bases:
        if len(bases) > 1 and _probe_base(base, api_key) is None:
            errors.append(f"{base}: unreachable")
            continue
        try:
            text, in_tok, out_tok, done = _generate_at(base, api_key, model_tag, prompt, max_tokens)
            return text, in_tok, out_tok, base, done
        except Exception as exc:  # noqa: BLE001 - every base gets its shot
            errors.append(f"{base}: {exc}")
    raise RuntimeError("Ollama bridge failed — " + "; ".join(errors))


def _dispatch_ssrf_ok(base: str) -> None:
    """DNS-rebinding / single-label guard at the moment of dispatch (Kimi #1+#5).
    IP literals were already range-checked in is_allowed_base; ollama.com and
    an explicitly-allowlisted host are intentional public targets. Every other
    hostname must resolve to a private/loopback/Tailscale IP right now."""
    host = (urllib.parse.urlparse(base).hostname or "").lower()
    if not host:
        raise RuntimeError(f"invalid Ollama base (no host): {base}")
    try:
        ipaddress.ip_address(host)
        return  # IP literal: already validated by is_allowed_base
    except ValueError:
        pass
    if host == "ollama.com" or host in _extra_allowed_hosts():
        return  # intentional public target
    if not _host_resolves_private(host):
        raise RuntimeError(f"SSRF guard: '{host}' did not resolve to a private/loopback/Tailscale IP")


def _generate_at(base: str, api_key: str, model_tag: str, prompt: str, max_tokens: int) -> tuple[str, int, int, Optional[str]]:
    _dispatch_ssrf_ok(base)
    payload = json.dumps({
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        # Thinking models (glm-5.2, qwen3.5, ...) otherwise spend the whole
        # num_predict budget on hidden reasoning and return an EMPTY response,
        # which the router misreads as a bridge failure (observed live
        # 2026-07-17). Ignored by non-thinking models.
        "think": False,
        "options": {"num_predict": max_tokens},
    }).encode("utf-8")
    request = urllib.request.Request(base + "/api/generate", data=payload, headers=_auth_headers(api_key))
    with urllib.request.urlopen(request, timeout=GENERATE_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = (body.get("response") or "").strip()
    if not text:
        raise ValueError(f"no response from Ollama: {str(body.get('error', body))[:200]}")
    # Ollama signals token-cap truncation via done_reason == "length"; surface
    # it so the one-strike escalation fires on a truncated bridge reply (Kimi #3).
    done_reason = "max_tokens" if body.get("done_reason") == "length" else None
    return text, int(body.get("prompt_eval_count") or 0), int(body.get("eval_count") or 0), done_reason


def _classify(task: str) -> str:
    if not anthropic_ready():
        # OFFLINE mode has exactly one engine, so classification is moot —
        # everything non-stakes runs on the bridge (stakes is refused later).
        if _ollama_configured():
            return "glm"
        raise RouterSetupError(
            "cannot classify: ANTHROPIC_API_KEY is unset and no Ollama bridge is "
            "configured or reachable. Run `python router.py --doctor`."
        )
    response = _client().messages.create(
        model=_model_id("haiku"),
        max_tokens=5,
        messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(task=task)}],
    )
    word = _response_text(response).strip().upper()
    return {
        "HAIKU": "haiku",
        "SONNET": "sonnet",
        "GLM": "glm",
        "OPUS": "opus",
        "FABLE": "fable",
    }.get(word, "sonnet")


def _normalise_tier(tier: Optional[str]) -> str:
    if tier is None:
        return "sonnet"
    value = tier.strip().lower()
    if value not in MODEL_REGISTRY:
        allowed = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"unknown tier {tier!r}; expected one of: {allowed}")
    return value


def _tier_enabled(tier: str) -> bool:
    return True


BENCH_REPORTS_DIR = pathlib.Path(__file__).parent / "bench" / "reports"
# A bench winner must pass EVERY probe it ran, and these two must be present —
# they encode the business rules (never invent a price; round tiers UP).
CRITICAL_PROBES = {"price-honesty", "tier-math"}


def bench_allocation() -> Optional[dict]:
    """Auto-allocation from the latest committed bench report: among bridge
    models with a clean sweep (all probes PASS, critical probes present),
    pick the lowest average latency. Returns {"model", "date", "avg_latency_s"}
    or None (no report / no qualifier / disabled).

    Weekly loop: model-bench.yml commits a fresh report Mondays -> any machine
    that pulls gets the new allocation automatically. Disable with
    CLAUDE_ROUTER_AUTO_ALLOCATE=0. Explicit env overrides always win.
    """
    if os.getenv("CLAUDE_ROUTER_AUTO_ALLOCATE", "1").strip().lower() in {"0", "false", "off"}:
        return None
    try:
        latest = max(BENCH_REPORTS_DIR.glob("*.json"))
    except (ValueError, OSError):
        return None
    try:
        report = json.loads(latest.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None

    best: Optional[dict] = None
    for model, row in (report.get("models") or {}).items():
        if not isinstance(row, dict) or row.get("baseline"):
            continue  # Claude baseline rows are comparison points, not allocatable
        probes = {k: v for k, v in row.items() if isinstance(v, dict) and "pass" in v}
        if not probes or not CRITICAL_PROBES.issubset(probes):
            continue
        if not all(v.get("pass") for v in probes.values()):
            continue
        lats = [v["latency_s"] for v in probes.values() if "latency_s" in v]
        avg = sum(lats) / len(lats) if lats else float("inf")
        if best is None or avg < best["avg_latency_s"]:
            best = {"model": model, "date": report.get("date", latest.stem),
                    "avg_latency_s": round(avg, 2)}
    return best


def _model_id(tier: str) -> str:
    """Resolve a tier's model id: CLAUDE_ROUTER_<TIER>_MODEL wins, then (for
    glm) GLM_OLLAMA_TAG, then the latest bench report's clean-sweep winner
    (auto-allocation), then the registry default."""
    explicit = os.getenv(f"CLAUDE_ROUTER_{tier.upper()}_MODEL", "").strip()
    if explicit:
        return explicit
    if tier == "glm":
        tag = os.getenv("GLM_OLLAMA_TAG", "").strip()
        if tag:
            return tag
        allocated = bench_allocation()
        if allocated:
            return allocated["model"]
    return MODEL_REGISTRY[tier].api_id


def _effort_for(tier: str, explicit_effort: Optional[str]) -> Optional[str]:
    if not MODEL_REGISTRY[tier].supports_effort:
        return None

    value = (
        explicit_effort
        or os.getenv(f"CLAUDE_ROUTER_{tier.upper()}_EFFORT")
        or os.getenv("CLAUDE_ROUTER_EFFORT")
        or "high"
    )
    value = value.strip().lower()

    if value not in SUPPORTED_EFFORT_LEVELS:
        allowed = ", ".join(sorted(SUPPORTED_EFFORT_LEVELS))
        raise ValueError(f"unsupported effort {value!r}; expected one of: {allowed}")
    return value


def _response_text(response: object) -> str:
    blocks = getattr(response, "content", []) or []
    texts = [getattr(block, "text", "") for block in blocks if getattr(block, "text", "")]
    return "\n".join(texts).strip()


_CSV_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: object) -> str:
    """Neutralise spreadsheet formula injection (Kimi review #6): a field
    beginning with =/+/-/@ executes as a formula when the CSV is opened in
    Excel/Sheets. csv.writer quotes commas but does NOT stop this. Prefix a
    single quote so the cell is treated as text. model_id/via can carry
    env-influenced values, so this matters."""
    s = str(value)
    if s.startswith(_CSV_FORMULA_TRIGGERS):
        return "'" + s
    return s


def _log(tier: str, model_id: str, effort: str, in_tok: int, out_tok: int, status: str,
         via: str = "") -> None:
    # Logging must NEVER break a dispatch that already spent an API call
    # (Kimi review #4): make the dir and swallow any write error.
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if is_new:
                writer.writerow(["timestamp_utc", "tier", "model_id", "effort",
                                 "input_tokens", "output_tokens", "status", "via"])
            writer.writerow([_csv_safe(v) for v in (
                dt.datetime.now(dt.timezone.utc).isoformat(),
                tier, model_id, effort, in_tok, out_tok, status, via,
            )])
    except Exception as exc:  # noqa: BLE001 - a log write must not crash a real dispatch
        print(f"[router] usage-log write failed: {exc}", file=sys.stderr, flush=True)


def last_dispatches(count: int = 10) -> list[dict]:
    """The most recent usage-log rows, newest last — 'what ran where'."""
    path = _log_path()
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows[-count:]


def _print_doctor_report(report: dict) -> None:
    width = max(len(str(row["check"])) for row in report["rows"])
    for row in report["rows"]:
        mark = "PASS" if row["ok"] else "FAIL"
        line = f"  [{mark}] {str(row['check']).ljust(width)}  {row['detail']}"
        print(line)
        if row["fix"]:
            print(f"         fix: {row['fix']}")
    print(f"\n  Claude tiers : {'READY' if report['claude_ready'] else 'OFFLINE'}")
    print(f"  Ollama bridge: {'READY via ' + str(report['bridge_base']) if report['bridge_ready'] else 'OFFLINE'}")
    print(f"  Router mode  : {report['mode']}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Claude model router v5.1 — multi-engine dispatch")
    parser.add_argument("task", nargs="*", help="the task/prompt to run")
    parser.add_argument("--tier", choices=list(MODEL_REGISTRY), help="manual tier override")
    parser.add_argument("--effort", choices=sorted(SUPPORTED_EFFORT_LEVELS), help="effort override")
    parser.add_argument("--stakes", action="store_true",
                        help="NUMBERS RULE: keep this task on Claude tiers only")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--doctor", action="store_true", help="run the setup check and exit")
    parser.add_argument("--registry", action="store_true", help="print the active registry as JSON")
    parser.add_argument("--last", type=int, nargs="?", const=10, metavar="N",
                        help="show the last N dispatches (what ran WHERE) and exit")
    args = parser.parse_args()

    if args.last is not None:
        rows = last_dispatches(args.last)
        if not rows:
            print("no dispatches logged yet")
        for row in rows:
            ts = row.get("timestamp_utc", "")[:19]
            print(f"{ts}  {row.get('tier', ''):7s} {row.get('model_id', ''):26s} "
                  f"{row.get('status', ''):16s} via {row.get('via') or '(pre-via log row)'}")
        sys.exit(0)
    if args.doctor:
        report = doctor()
        _print_doctor_report(report)
        sys.exit(0 if report["ok"] else 1)
    if args.registry:
        print(json.dumps(registry(), indent=2))
        sys.exit(0)

    prompt = " ".join(args.task) or "Hello! What can you do?"
    print(run(prompt, max_tokens=args.max_tokens, tier=args.tier, effort=args.effort, stakes=args.stakes))
