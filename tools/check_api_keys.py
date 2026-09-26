#!/usr/bin/env python3
"""Validate every provider API key with the smallest possible calls.

For each provider whose key is set: list its models (free), then ask for one
tiny reply (a few tokens) from a model id already proven on this account.
Prints one row per provider: ok, fail (with the HTTP status and the provider's
own error text, key redacted), or skipped (key not set). Never prints a key.

Exit code is non-zero if any key that IS set fails, so a green run means every
configured key works. An unset key is reported, not failed.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

TIMEOUT = 30


def _env(name, default=""):
    # CI passes an unset secret as an empty string, so empty means "use the default".
    return os.getenv(name, "").strip() or default


def _redact(text):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "QWEN_API_KEY", "OLLAMA_API_KEY",
                 "XAI_API_KEY"):
        secret = _env(name)
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _call(url, headers, body=None):
    """One request, retried up to 3 times when the provider says it is busy (429/503)."""
    for attempt in range(3):
        status, payload = _call_once(url, headers, body)
        if status not in (429, 503) or attempt == 2:
            return status, payload
        time.sleep(2 * (attempt + 1))
    return status, payload


def _call_once(url, headers, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={**headers, "content-type": "application/json"},
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, {"error": _redact(err.read().decode("utf-8", "replace"))[:300]}
    except Exception as err:  # network failure is a failure, never a pass
        return 0, {"error": _redact(f"{type(err).__name__}: {err}")[:300]}


def _openai_compat(base, key, model, extra_headers=None, token_field="max_tokens"):
    headers = {"authorization": f"Bearer {key}", **(extra_headers or {})}
    status, body = _call(f"{base.rstrip('/')}/models", headers)
    if status != 200:
        return False, f"list models HTTP {status}: {body.get('error', '')}"
    ids = [m.get("id", "") for m in body.get("data", []) if isinstance(m, dict)]
    count = len(ids)
    if not model:
        # No model proven on this account yet: use the first id the provider
        # itself lists, rather than guessing one.
        chat_ids = [i for i in ids if "embed" not in i and "image" not in i and "tts" not in i]
        if not chat_ids:
            return False, f"{count} models listed; none usable for a reply check"
        model = chat_ids[0]
    status, body = _call(f"{base.rstrip('/')}/chat/completions", headers,
                         {"model": model, token_field: 16,
                          "messages": [{"role": "user", "content": "Reply with the word ok."}]})
    if status in (429, 503):
        # The key authenticated (the model list worked); the model itself is busy.
        return True, f"{count} models listed; key valid, but {model} was busy (HTTP {status}) after 3 tries"
    if status != 200:
        return False, f"{count} models listed; reply from {model} HTTP {status}: {body.get('error', '')}"
    return True, f"{count} models listed; {model} replied"


def check_anthropic():
    key = _env("ANTHROPIC_API_KEY")
    if not key:
        return None
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if _env("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = _env("ANTHROPIC_WORKSPACE_ID")
    model = _env("ANTHROPIC_CHECK_MODEL", "claude-haiku-4-5")
    status, body = _call("https://api.anthropic.com/v1/models", headers)
    if status != 200:
        hint = ""
        if "not scoped to a workspace" in json.dumps(body):
            hint = " -> use a workspace-scoped key or set ANTHROPIC_WORKSPACE_ID"
        return False, f"list models HTTP {status}: {body.get('error', '')}{hint}"
    status, body = _call("https://api.anthropic.com/v1/messages", headers,
                         {"model": model, "max_tokens": 8,
                          "messages": [{"role": "user", "content": "Reply with the word ok."}]})
    if status != 200:
        return False, f"reply from {model} HTTP {status}: {body.get('error', '')}"
    return True, f"{model} replied"


def check_openai():
    key = _env("OPENAI_API_KEY")
    if not key:
        return None
    return _openai_compat(_env("OPENAI_BASE_URL", "https://api.openai.com/v1"), key,
                          _env("OPENAI_CHECK_MODEL", "gpt-5.6-sol"), token_field="max_completion_tokens")


def check_gemini():
    key = _env("GEMINI_API_KEY")
    if not key:
        return None
    return _openai_compat(_env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
                          key, _env("GEMINI_CHECK_MODEL", "gemini-3.8-flash"))


def check_qwen():
    key = _env("QWEN_API_KEY")
    if not key:
        return None
    base = _env("QWEN_BASE_URL")
    if not base:
        return False, "QWEN_API_KEY is set but QWEN_BASE_URL is not (the workspace endpoint is required)"
    return _openai_compat(base, key, _env("QWEN_CHECK_MODEL", "qwen-turbo"))


def check_ollama():
    key = _env("OLLAMA_API_KEY")
    if not key:
        return None
    headers = {"authorization": f"Bearer {key}"}
    model = _env("OLLAMA_CHECK_MODEL", "deepseek-v4.1-flash")
    status, body = _call("https://ollama.com/api/tags", headers)
    if status != 200:
        return False, f"list models HTTP {status}: {body.get('error', '')}"
    count = len(body.get("models", []))
    status, body = _call("https://ollama.com/api/chat", headers,
                         {"model": model, "stream": False, "think": False,
                          "messages": [{"role": "user", "content": "Reply with the word ok."}],
                          "options": {"num_predict": 16}})
    if status != 200:
        return False, f"{count} models listed; reply from {model} HTTP {status}: {body.get('error', '')}"
    return True, f"{count} models listed; {model} replied"


def check_grok():
    key = _env("XAI_API_KEY")
    if not key:
        return None
    return _openai_compat(_env("XAI_BASE_URL", "https://api.x.ai/v1"), key, _env("XAI_CHECK_MODEL"))


CHECKS = [("Anthropic", check_anthropic), ("OpenAI", check_openai), ("Gemini", check_gemini),
          ("Qwen", check_qwen), ("Ollama Cloud", check_ollama),
          ("Grok (xAI)", check_grok)]

SUBSCRIPTION_TOKENS = [("Claude subscription (CLAUDE_CODE_OAUTH_TOKEN)", "CLAUDE_CODE_OAUTH_TOKEN")]


def main():
    rows, failed = [], False
    for name, fn in CHECKS:
        result = fn()
        if result is None:
            rows.append((name, "skipped", "key not set"))
            continue
        ok, detail = result
        failed |= not ok
        rows.append((name, "ok" if ok else "FAIL", detail))
    for name, var in SUBSCRIPTION_TOKENS:
        # Presence only: a subscription token is for Claude Code itself, so it is
        # not exercised against the API here.
        rows.append((name, "present" if _env(var) else "not set", "presence check only"))

    lines = ["| Provider | Result | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {r} | {d.replace('|', '/')} |" for n, r, d in rows]
    report = "\n".join(lines)
    print(report)
    summary = _env("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## API key check\n\n" + report + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
