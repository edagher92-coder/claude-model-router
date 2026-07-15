"""Ollama worker caller — GLM 5.2 (and future non-Claude tiers) via the
account's Ollama bridge conventions (see claude-defaults/tools/ollama_route.py).

Structured output: Ollama's /api/chat accepts a JSON Schema as `format`,
so the worker reply parses as a result envelope just like the Anthropic
forced-tool-choice path.

Env:  CLAUDE_ROUTER_OLLAMA_URL  (default http://localhost:11434; with only
      OLLAMA_API_KEY set, defaults to https://ollama.com — Ollama Cloud)
      OLLAMA_API_KEY            (Cloud auth, optional)
      GLM_OLLAMA_TAG            (override the model tag; default glm-5.2:cloud)

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

_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()
_DEFAULT_BASE = "https://ollama.com" if _API_KEY else "http://localhost:11434"
BASE = os.environ.get("CLAUDE_ROUTER_OLLAMA_URL", _DEFAULT_BASE).rstrip("/")

# envelope tag -> Ollama model tag
OLLAMA_TAGS = {
    "glm-5.2": os.environ.get("GLM_OLLAMA_TAG", "glm-5.2:cloud"),
}


def is_ollama_model(model: str) -> bool:
    return model in OLLAMA_TAGS


def call(model: str, system: str, message: str, submit_tool: dict, timeout: int = 600) -> dict:
    """Match the ModelCaller signature used by core.delegate."""
    payload = json.dumps({
        "model": OLLAMA_TAGS[model],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        "stream": False,
        "format": submit_tool["input_schema"],
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if _API_KEY:
        headers["Authorization"] = "Bearer " + _API_KEY

    last_error: Exception | None = None
    for delay in (0,) + RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(BASE + "/api/chat", data=payload, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            result = parse_chat_response(body)
            result["usage_note"] = (
                f"ollama:{OLLAMA_TAGS[model]} "
                f"prompt_tokens={body.get('prompt_eval_count', '?')} "
                f"output_tokens={body.get('eval_count', '?')}"
            )
            return result
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise RuntimeError(
                    f"Ollama rejected the request (HTTP {exc.code}) — check the model tag "
                    f"'{OLLAMA_TAGS[model]}' and OLLAMA_API_KEY; not retrying"
                ) from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(
        f"Ollama worker at {BASE} failed after {len(RETRY_DELAYS) + 1} attempts: {last_error}. "
        "If the bridge is down, re-route this task to claude-sonnet-5 per the fallback rule."
    )


def parse_chat_response(body: dict) -> dict:
    """Extract and parse the structured envelope from an /api/chat body."""
    content = (body.get("message") or {}).get("content", "")
    if not content:
        raise ValueError(f"Ollama response had no message content: {str(body)[:200]}")
    return json.loads(content)
