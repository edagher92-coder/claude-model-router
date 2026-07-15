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

Usage:
    from router import run
    reply = run("Refactor this function", tier="sonnet", effort="high")

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY in env.

Logs every dispatch to router-usage.csv.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Optional

import anthropic

client = anthropic.Anthropic()

LOG = pathlib.Path(__file__).parent / os.getenv("CLAUDE_ROUTER_LOG", "router-usage.csv")


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
            tier is skipped in classification and escalation.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    current_tier = _normalise_tier(tier) if tier else _classify(task)
    current_tier = _apply_stakes(current_tier, stakes)
    last_text = ""

    for _attempt in range(5):
        model_id = _model_id(current_tier)
        requested_effort = _effort_for(current_tier, effort)

        if MODEL_REGISTRY[current_tier].engine == "ollama":
            try:
                text, input_tokens, output_tokens = _ollama_generate(model_id, task, max_tokens)
            except Exception:
                # Bridge down or model tag missing: back to Claude, never block.
                _log(current_tier, model_id, "", 0, 0, "fallback_bridge")
                current_tier = FALLBACK[current_tier]
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
                response = client.messages.create(**request)
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

        last_text = text
        _log(current_tier, model_id, requested_effort or "", input_tokens, output_tokens, "ok")

        # Guardrail: empty/very short responses often mean the tier was too weak or the task needs more capability.
        if len(text.strip()) < 20 and current_tier != "fable":
            current_tier = _apply_stakes(ESCALATE[current_tier], stakes)
            continue

        return text

    return last_text


def _apply_stakes(tier: str, stakes: bool) -> str:
    """NUMBERS RULE enforcement: stakes work never lands on the Ollama bridge."""
    if stakes and MODEL_REGISTRY[tier].engine == "ollama":
        return "sonnet"
    return tier


def _ollama_generate(model_tag: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
    """Dispatch to the Ollama bridge using the account's shared conventions
    (see claude-defaults/tools/ollama_route.py and hq_orchestrator/ollama_caller)."""
    import urllib.request

    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    default_base = "https://ollama.com" if api_key else "http://localhost:11434"
    base = os.getenv("CLAUDE_ROUTER_OLLAMA_URL", default_base).rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    import json as _json
    payload = _json.dumps({
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }).encode("utf-8")
    request = urllib.request.Request(base + "/api/generate", data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        body = _json.loads(response.read().decode("utf-8"))
    text = (body.get("response") or "").strip()
    if not text:
        raise ValueError(f"no response from Ollama: {str(body.get('error', body))[:200]}")
    return text, int(body.get("prompt_eval_count") or 0), int(body.get("eval_count") or 0)


def _classify(task: str) -> str:
    response = client.messages.create(
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
    env_name = f"CLAUDE_ROUTER_{tier.upper()}_MODEL"
    return os.getenv(env_name, MODEL_REGISTRY[tier].api_id).strip()


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
    is_new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as file:
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


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or "Hello! What can you do?"
    print(run(prompt))
