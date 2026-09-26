"""Subscription-first auth for Elie's LOCAL, interactive tools (stdlib only).

Resolution order for Claude (``resolve_claude_auth``):

1. ``CI`` set (any truthy value)      -> API key, always. CI never uses a login.
2. ``ROUTER_AUTH=key``                -> API key (the override for servers).
3. Claude Code CLI signed in          -> subscription (via ``claude -p``).
4. ``ROUTER_AUTH=subscription``       -> none (no silent key fallback).
5. ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` set -> API key.
6. otherwise                          -> none.

Why a subprocess and not the token: Anthropic's terms allow subscription
OAuth only for "ordinary use of Claude Code and other native Anthropic
applications" (code.claude.com/docs/en/legal-and-compliance). So the
subscription path runs the unmodified ``claude`` binary in print mode and
never reads, copies or forwards its credential. The API key path is untouched.

Servers (a routing box, anything answering customers) must set
``ROUTER_AUTH=key``. Nothing here stores or prints a secret.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Mapping, Optional

SUBSCRIPTION = "subscription"
KEY = "key"
NONE = "none"

OVERRIDE_ENV = "ROUTER_AUTH"
OVERRIDE_VALUES = ("auto", "key", "subscription")
CLAUDE_KEY_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

# `claude -p` always uses ANTHROPIC_API_KEY when it is present (auth precedence,
# code.claude.com/docs/en/authentication), and ANTHROPIC_AUTH_TOKEN outranks it.
# Strip both from the child so the subscription login is what actually runs.
_CLAUDE_CHILD_STRIP = CLAUDE_KEY_ENV
_CODEX_CHILD_STRIP = ("OPENAI_API_KEY", "CODEX_API_KEY")

PROBE_TIMEOUT = 20
CLI_TIMEOUT = int(os.getenv("ROUTER_CLI_TIMEOUT", "600") or "600")


class ClaudeCLIError(RuntimeError):
    """`claude -p` did not return a usable answer. The message never holds a secret."""


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no", "off")


def in_ci(env: Optional[Mapping[str, str]] = None) -> bool:
    return _truthy(_env(env).get("CI"))


def auth_override(env: Optional[Mapping[str, str]] = None) -> str:
    value = (_env(env).get(OVERRIDE_ENV) or "").strip().lower() or "auto"
    if value not in OVERRIDE_VALUES:
        raise ValueError(f"{OVERRIDE_ENV}={value!r} — use one of {', '.join(OVERRIDE_VALUES)}")
    return value


def keys_forced(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when every provider must use its API key (CI, or ROUTER_AUTH=key)."""
    return in_ci(env) or auth_override(env) == "key"


def has_claude_key(env: Optional[Mapping[str, str]] = None) -> bool:
    e = _env(env)
    return any((e.get(name) or "").strip() for name in CLAUDE_KEY_ENV)


def child_env(strip: tuple, env: Optional[Mapping[str, str]] = None) -> dict:
    return {k: v for k, v in _env(env).items() if k not in strip}


# --------------------------------------------------------------------------- #
# Claude Code CLI login probe (local command, no model call, cached per process)
# --------------------------------------------------------------------------- #
_LOGIN_CACHE: dict = {}


def claude_cli() -> Optional[str]:
    return shutil.which("claude")


def claude_login_status() -> tuple[bool, str]:
    """(signed_in, detail) from `claude auth status --json`, run with the API-key
    variables removed so a key cannot masquerade as a login."""
    cli = claude_cli()
    if not cli:
        return False, "claude CLI not on PATH"
    if cli in _LOGIN_CACHE:
        return _LOGIN_CACHE[cli]
    try:
        proc = subprocess.run(
            [cli, "auth", "status", "--json"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT,
            env=child_env(_CLAUDE_CHILD_STRIP), stdin=subprocess.DEVNULL,
        )
        data = json.loads(proc.stdout or "{}")
        signed_in = bool(data.get("loggedIn"))
        detail = f"claude auth status: loggedIn={signed_in}, authMethod={data.get('authMethod', '?')}"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        signed_in, detail = False, f"claude auth status failed ({type(exc).__name__})"
    _LOGIN_CACHE[cli] = (signed_in, detail)
    return signed_in, detail


def resolve_claude_auth(env: Optional[Mapping[str, str]] = None,
                        signed_in: Optional[bool] = None) -> str:
    """Return SUBSCRIPTION, KEY or NONE. `signed_in` injects the CLI probe (tests)."""
    key = has_claude_key(env)
    if keys_forced(env):
        return KEY if key else NONE
    if signed_in is None:
        signed_in = claude_login_status()[0]
    if signed_in:
        return SUBSCRIPTION
    if auth_override(env) == "subscription":
        return NONE
    return KEY if key else NONE


# --------------------------------------------------------------------------- #
# The subscription call: the unmodified `claude` binary in print mode
# --------------------------------------------------------------------------- #
def claude_print(prompt: str, model: str, system: Optional[str] = None,
                 effort: Optional[str] = None, json_schema: Optional[dict] = None,
                 timeout: Optional[int] = None) -> dict:
    """Run one `claude -p` turn with every built-in tool disabled and return the
    parsed `--output-format json` result. The prompt goes over stdin."""
    cli = claude_cli()
    if not cli:
        raise ClaudeCLIError("claude CLI not on PATH")
    args = [cli, "-p", "--model", model, "--output-format", "json",
            "--tools", "", "--no-session-persistence"]
    if system:
        args += ["--system-prompt", system]
    if effort:
        args += ["--effort", effort]
    if json_schema is not None:
        args += ["--json-schema", json.dumps(json_schema)]
    try:
        proc = subprocess.run(
            args, input=prompt, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout or CLI_TIMEOUT,
            env=child_env(_CLAUDE_CHILD_STRIP), cwd=tempfile.gettempdir(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ClaudeCLIError(f"claude -p could not run ({type(exc).__name__})") from exc
    try:
        data = json.loads(proc.stdout or "")
    except ValueError:
        data = None
    if proc.returncode != 0 or not isinstance(data, dict) or data.get("is_error"):
        why = (data or {}).get("subtype") if isinstance(data, dict) else None
        tail = (proc.stderr or "").strip()[-300:]
        raise ClaudeCLIError(f"claude -p exited {proc.returncode} ({why or 'no result'}) {tail}".strip())
    return data


def as_message(data: dict) -> SimpleNamespace:
    """Shape a `claude -p` JSON result like an SDK Message (content/usage/stop_reason)."""
    usage = data.get("usage") or {}
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=str(data.get("result") or ""))],
        usage=SimpleNamespace(input_tokens=int(usage.get("input_tokens") or 0),
                              output_tokens=int(usage.get("output_tokens") or 0)),
        stop_reason=data.get("stop_reason"),
        structured_output=data.get("structured_output"),
        via="Anthropic via Claude Code CLI (subscription)",
    )


# --------------------------------------------------------------------------- #
# Per-provider report for `router.py doctor` (local checks only, no model calls)
# --------------------------------------------------------------------------- #
def _home() -> pathlib.Path:
    return pathlib.Path.home()


def _codex_status() -> tuple[bool, str]:
    cli = shutil.which("codex")
    if not cli:
        return False, "codex CLI not on PATH"
    try:
        proc = subprocess.run([cli, "login", "status"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT,
                              env=child_env(_CODEX_CHILD_STRIP), stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"codex login status failed ({type(exc).__name__})"
    # Source of truth: codex-rs/cli/src/login.rs prints "Logged in using ChatGPT"
    # (exit 0) for a ChatGPT-plan login. Any other mode is not the subscription.
    said = (proc.stderr or "") + (proc.stdout or "")
    return proc.returncode == 0 and "Logged in using ChatGPT" in said, "codex login status"


def provider_auth(env: Optional[Mapping[str, str]] = None) -> list[dict]:
    """One row per provider: {"provider", "auth": subscription|key|none, "detail"}.
    Subscription detection is local (CLI status command or its documented
    credential file); it never makes a model call and never reads a secret."""
    e = _env(env)
    forced = keys_forced(e)

    def key_row(provider: str, names: tuple, sub_detail: str = "") -> dict:
        present = [n for n in names if (e.get(n) or "").strip()]
        if present:
            return {"provider": provider, "auth": KEY, "detail": f"{present[0]} set"
                    + (f"; {sub_detail}" if sub_detail else "")}
        return {"provider": provider, "auth": NONE, "detail": sub_detail or "no login, no key"}

    why_forced = "CI" if in_ci(e) else f"{OVERRIDE_ENV}=key"
    rows = []

    # Claude — the only provider the router itself dispatches to.
    mode = resolve_claude_auth(e)
    login_detail = f"keys forced ({why_forced})" if forced else claude_login_status()[1]
    if mode == SUBSCRIPTION:
        rows.append({"provider": "Claude", "auth": SUBSCRIPTION,
                     "detail": "Claude Code login, dispatched via `claude -p` — " + login_detail})
    elif mode == KEY:
        rows.append(key_row("Claude", CLAUDE_KEY_ENV, login_detail))
    else:
        rows.append({"provider": "Claude", "auth": NONE, "detail": login_detail
                     + (f"; {OVERRIDE_ENV}=subscription allows no key fallback"
                        if auth_override(e) == "subscription" else "")})

    checks = [
        ("OpenAI", ("OPENAI_API_KEY",), _codex_status, "Codex CLI signed in with ChatGPT"),
        ("xAI Grok", ("XAI_API_KEY",),
         lambda: (bool(shutil.which("grok")) and (_home() / ".grok" / "auth.json").is_file(),
                  "grok CLI + ~/.grok/auth.json"),
         "Grok Build CLI signed in (SuperGrok)"),
        ("Gemini", ("GEMINI_API_KEY",),
         lambda: (bool(shutil.which("gemini")) and (_home() / ".gemini" / "oauth_creds.json").is_file(),
                  "gemini CLI + ~/.gemini/oauth_creds.json"),
         "Gemini CLI signed in with Google (AI Pro limits)"),
        ("Qwen", ("QWEN_API_KEY",),
         lambda: (bool(shutil.which("qwen")) and (
             bool((e.get("BAILIAN_CODING_PLAN_API_KEY") or "").strip())
             or "BAILIAN_CODING_PLAN_API_KEY" in _read_text(_home() / ".qwen" / "settings.json")),
             "qwen CLI + Coding Plan key configured"),
         "Qwen Code with the Coding Plan"),
    ]
    for provider, names, probe, label in checks:
        if forced:
            rows.append(key_row(provider, names, f"keys forced ({why_forced})"))
            continue
        ok, how = probe()
        if ok:
            rows.append({"provider": provider, "auth": SUBSCRIPTION, "detail": f"{label} ({how})"})
        else:
            rows.append(key_row(provider, names, f"no subscription login ({how})"))
    return rows


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
