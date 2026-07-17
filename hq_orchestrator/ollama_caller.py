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

import json
import os
import time
import urllib.error
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


def _bases() -> list[str]:
    """Priority-ordered base URLs — keep in step with router._ollama_bases."""
    key = _api_key()
    raw = os.environ.get("CLAUDE_ROUTER_OLLAMA_URL", "").strip()
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        urls = ["https://ollama.com" if key else "http://localhost:11434"]
    if key and "https://ollama.com" not in urls:
        urls.append("https://ollama.com")
    deduped: list[str] = []
    for url in urls:
        if url not in deduped:
            deduped.append(url)
    return deduped


def _tag(model: str) -> str:
    return os.environ.get(_TAG_ENV[model], "").strip() or _DEFAULT_TAGS[model]


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
