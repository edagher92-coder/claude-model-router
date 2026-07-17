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
import json
import os
import pathlib
import urllib.error
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
        if url not in seen:
            seen.add(url)
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
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    claude_ok = anthropic_ready()
    current_tier = _normalise_tier(tier) if tier else _classify(task)
    current_tier = _apply_stakes(current_tier, stakes)
    if not claude_ok:
        current_tier = _reroute_offline(current_tier, stakes)
    tried: set[str] = set()
    last_text = ""

    for _attempt in range(len(LADDER)):
        tried.add(current_tier)
        model_id = _model_id(current_tier)
        requested_effort = _effort_for(current_tier, effort)
        stop_reason: Optional[str] = None

        if MODEL_REGISTRY[current_tier].engine == "ollama":
            try:
                text, input_tokens, output_tokens = _ollama_generate(model_id, task, max_tokens)
            except Exception as exc:
                # Bridge down or tag missing = infra failure, not capability:
                # recover on Sonnet if we haven't already tried it, otherwise
                # keep climbing. The tried-set prevents sonnet<->glm ping-pong.
                _log(current_tier, model_id, "", 0, 0, "fallback_bridge")
                recovery = (
                    "sonnet"
                    if claude_ok and "sonnet" not in tried
                    else _next_tier_up(current_tier, tried, stakes, claude_ok)
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
                response = _client().messages.create(**request)
            except anthropic.APIStatusError as exc:
                # Fable can be unavailable or permission-restricted on some accounts.
                # Fall back one tier for access/availability errors, but re-raise all other errors.
                if getattr(exc, "status_code", None) in {400, 401, 403, 404} and current_tier == "fable":
                    _log(current_tier, model_id, requested_effort or "", 0, 0, f"fallback_{exc.status_code}")
                    current_tier = FALLBACK[current_tier]
                    continue
                raise
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            text = _response_text(response)
            stop_reason = getattr(response, "stop_reason", None)

        last_text = text

        # Policy hard rule: ONE failed or incomplete response at a tier =>
        # escalate immediately (empty/short, token-cap truncation, refusal).
        if _is_incomplete(text, stop_reason):
            next_tier = _next_tier_up(current_tier, tried, stakes, claude_ok)
            _log(current_tier, model_id, requested_effort or "", input_tokens, output_tokens,
                 "escalated" if next_tier else "incomplete_at_top")
            if next_tier is None:
                return text  # top of ladder: surface what we have, honestly logged
            current_tier = next_tier
            continue

        _log(current_tier, model_id, requested_effort or "", input_tokens, output_tokens, "ok")
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
    """True when a reply should trigger the one-strike escalation: empty or
    very short, truncated by the token cap, or an outright refusal."""
    stripped = text.strip()
    if len(stripped) < 20:
        return True
    if stop_reason == "max_tokens":
        return True
    return stripped.lower().startswith(REFUSAL_PREFIXES)


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


def _ollama_generate(model_tag: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
    """Dispatch to the first reachable base in the Ollama chain (routing
    server / local daemon first, then Ollama Cloud — see _ollama_bases)."""
    bases = _ollama_bases()
    errors: list[str] = []
    for base, api_key in bases:
        if len(bases) > 1 and _probe_base(base, api_key) is None:
            errors.append(f"{base}: unreachable")
            continue
        try:
            return _generate_at(base, api_key, model_tag, prompt, max_tokens)
        except Exception as exc:  # noqa: BLE001 - every base gets its shot
            errors.append(f"{base}: {exc}")
    raise RuntimeError("Ollama bridge failed — " + "; ".join(errors))


def _generate_at(base: str, api_key: str, model_tag: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
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
    return text, int(body.get("prompt_eval_count") or 0), int(body.get("eval_count") or 0)


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


def _model_id(tier: str) -> str:
    """Resolve a tier's model id: CLAUDE_ROUTER_<TIER>_MODEL wins, then (for
    glm) the account-wide GLM_OLLAMA_TAG convention, then the registry."""
    explicit = os.getenv(f"CLAUDE_ROUTER_{tier.upper()}_MODEL", "").strip()
    if explicit:
        return explicit
    if tier == "glm":
        tag = os.getenv("GLM_OLLAMA_TAG", "").strip()
        if tag:
            return tag
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


def _log(tier: str, model_id: str, effort: str, in_tok: int, out_tok: int, status: str) -> None:
    path = _log_path()
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(["timestamp_utc", "tier", "model_id", "effort", "input_tokens", "output_tokens", "status"])
        writer.writerow([
            dt.datetime.now(dt.timezone.utc).isoformat(),
            tier,
            model_id,
            effort,
            in_tok,
            out_tok,
            status,
        ])


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
    args = parser.parse_args()

    if args.doctor:
        report = doctor()
        _print_doctor_report(report)
        sys.exit(0 if report["ok"] else 1)
    if args.registry:
        print(json.dumps(registry(), indent=2))
        sys.exit(0)

    prompt = " ".join(args.task) or "Hello! What can you do?"
    print(run(prompt, max_tokens=args.max_tokens, tier=args.tier, effort=args.effort, stakes=args.stakes))
