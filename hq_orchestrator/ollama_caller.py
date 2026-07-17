"""Ollama worker caller — GLM 5.2 (and future non-Claude tiers) via the
account's Ollama bridge conventions (see claude-defaults/tools/ollama_route.py).

Structured output: Ollama's /api/chat accepts a JSON Schema as `format`,
so the worker reply parses as a result envelope just like the Anthropic
forced-tool-choice path.

Env (read at CALL time, so a long-lived server picks up changes/rotation):
  CLAUDE_ROUTER_OLLAMA_URL  one URL or a comma-separated priority list
                            (e.g. tailnet routing server, then a second
                            PC's daemon). Default http://localhost:11434;
                            with only OLLAMA_API_KEY set, https://ollama.com
                            (Ollama Cloud) is appended to the chain.
  OLLAMA_API_KEY            Cloud auth, optional.
  GLM_OLLAMA_TAG            override the model tag; default glm-5.2:cloud.

NUMBERS RULE (policy, enforced upstream by the orchestrator's routing):
never dispatch customer-facing price/quote/invoice/legal tasks here.
"""

from __future__ import annotations

import ipaddress
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

RETRY_DELAYS = (2, 4, 8, 16)

# envelope tag -> default Ollama model tag (env override read at call time)
_DEFAULT_TAGS = {
    "glm-5.2": "glm-5.2:cloud",
}
_TAG_ENV = {
    "glm-5.2": "GLM_OLLAMA_TAG",
}


def _api_key() -> str:
    return os.environ.get("OLLAMA_API_KEY", "").strip()


# --- SSRF guard: identical policy to router.is_allowed_base. Duplicated (not
# imported) because this package is stdlib-only and must not depend on router.py.
_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_HOSTNAMES = {"localhost", "ollama.com"}


def is_allowed_base(url: str) -> tuple[bool, str]:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:
        return False, f"unparseable URL: {exc}"
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"scheme '{parsed.scheme}' not allowed"
    host = parsed.hostname
    if not host:
        return False, "no host"
    host_l = host.lower()
    extra = {h.strip().lower() for h in os.environ.get("CLAUDE_ROUTER_OLLAMA_ALLOW_HOSTS", "").split(",") if h.strip()}
    if host_l in extra:
        return True, "allowlisted"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_loopback:
            return True, "loopback"
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return True, "tailscale"
        if ip.is_link_local:
            return False, "link-local/metadata blocked (SSRF)"
        if ip.is_private:
            return True, "private"
        return False, f"public IP {host} blocked (SSRF)"
    if host_l in _ALLOWED_HOSTNAMES or host_l.endswith(".ts.net"):
        return True, "allowed hostname"
    if "." not in host_l:                           # single-label = LAN, not public
        return True, "single-label LAN hostname"
    return False, f"public host '{host}' blocked (SSRF)"


def _bases() -> list[str]:
    """Priority-ordered base URLs — keep in step with router._ollama_bases.
    SSRF-filtered: a rejected base is dropped with a warning, never dispatched."""
    key = _api_key()
    raw = os.environ.get("CLAUDE_ROUTER_OLLAMA_URL", "").strip()
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        urls = ["https://ollama.com" if key else "http://localhost:11434"]
    if key and "https://ollama.com" not in urls:
        urls.append("https://ollama.com")
    deduped: list[str] = []
    for url in urls:
        if url in deduped:
            continue
        ok, reason = is_allowed_base(url)
        if not ok:
            print(f"[ollama_caller] BLOCKED base {url!r}: {reason}", file=sys.stderr, flush=True)
            continue
        deduped.append(url)
    return deduped


def _tag(model: str) -> str:
    """Resolve the Ollama model tag. Precedence matches the router so the two
    never disagree: GLM_OLLAMA_TAG env > router bench auto-allocation >
    registry default. The bench winner (e.g. kimi-k2.7-code on a strong week)
    is used verbatim as the tag."""
    env_tag = os.environ.get(_TAG_ENV[model], "").strip()
    if env_tag:
        return env_tag
    if model == "glm-5.2":
        allocated = _bench_allocated_tag()
        if allocated:
            return allocated
    return _DEFAULT_TAGS[model]


def _bench_allocated_tag() -> str:
    """The router's current bench allocation, if importable. Guarded so the
    orchestrator still works standalone (returns '' -> caller uses the default)."""
    try:
        import router  # sibling module at repo root
        alloc = router.bench_allocation()
        return alloc["model"] if alloc else ""
    except Exception:
        return ""


def is_ollama_model(model: str) -> bool:
    return model in _DEFAULT_TAGS


def call(model: str, system: str, message: str, submit_tool: dict, timeout: int = 600) -> dict:
    """Match the ModelCaller signature used by core.delegate."""
    tag = _tag(model)
    payload = json.dumps({
        "model": tag,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        "stream": False,
        # Thinking models otherwise burn the output budget on hidden reasoning
        # and can return an empty message (see router.py _generate_at).
        "think": False,
        "format": submit_tool["input_schema"],
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if _api_key():
        headers["Authorization"] = "Bearer " + _api_key()

    bases = _bases()
    last_error: Exception | None = None
    for delay in (0,) + RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        # Walk the chain each round: routing server first, cloud as backstop.
        for base in bases:
            request = urllib.request.Request(base + "/api/chat", data=payload, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                result = parse_chat_response(body)
                result["usage_note"] = (
                    f"ollama:{tag}@{base} "
                    f"prompt_tokens={body.get('prompt_eval_count', '?')} "
                    f"output_tokens={body.get('eval_count', '?')}"
                )
                return result
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500:
                    raise RuntimeError(
                        f"Ollama at {base} rejected the request (HTTP {exc.code}) — check the "
                        f"model tag '{tag}' and OLLAMA_API_KEY; not retrying"
                    ) from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
    raise RuntimeError(
        f"Ollama worker failed after {len(RETRY_DELAYS) + 1} attempts across {bases}: "
        f"{last_error}. If the bridge is down, re-route this task to claude-sonnet-5 "
        "per the fallback rule."
    )


def parse_chat_response(body: dict) -> dict:
    """Extract and parse the structured envelope from an /api/chat body."""
    content = (body.get("message") or {}).get("content", "")
    if not content:
        raise ValueError(f"Ollama response had no message content: {str(body)[:200]}")
    return json.loads(content)
