"""Subscription-first Claude auth: the resolution order, the `claude -p`
subprocess contract, the router wiring and the doctor report.

No test runs a real CLI or spends a token: subprocess and the login probe are
faked. conftest.py pins ROUTER_AUTH=key for every other suite; tests here
opt back in by setting ROUTER_AUTH themselves.
"""
import importlib
import json
import subprocess
import sys
import types

import pytest

import subscription_auth as sa

REAL_CLAUDE_PRINT = sa.claude_print  # captured before conftest's refusal stub

KEY = {"ANTHROPIC_API_KEY": "test-key"}


# --------------------------------------------------------------------------- #
# Resolution order
# --------------------------------------------------------------------------- #
def test_subscription_wins_over_a_key_when_signed_in():
    assert sa.resolve_claude_auth({**KEY}, signed_in=True) == sa.SUBSCRIPTION
    assert sa.resolve_claude_auth({}, signed_in=True) == sa.SUBSCRIPTION


def test_key_is_the_fallback_when_no_login():
    assert sa.resolve_claude_auth({**KEY}, signed_in=False) == sa.KEY
    assert sa.resolve_claude_auth({"ANTHROPIC_AUTH_TOKEN": "t"}, signed_in=False) == sa.KEY


def test_nothing_available_is_none():
    assert sa.resolve_claude_auth({}, signed_in=False) == sa.NONE
    assert sa.resolve_claude_auth({"ANTHROPIC_API_KEY": "   "}, signed_in=False) == sa.NONE


@pytest.mark.parametrize("ci", ["true", "1", "yes", "TRUE"])
def test_ci_forces_the_key_even_when_signed_in(ci):
    assert sa.resolve_claude_auth({"CI": ci, **KEY}, signed_in=True) == sa.KEY
    assert sa.resolve_claude_auth({"CI": ci}, signed_in=True) == sa.NONE
    # CI outranks an explicit subscription override too.
    assert sa.resolve_claude_auth({"CI": ci, "ROUTER_AUTH": "subscription", **KEY},
                                  signed_in=True) == sa.KEY


@pytest.mark.parametrize("ci", ["", "0", "false", "no", "off"])
def test_falsy_ci_values_are_not_ci(ci):
    assert sa.resolve_claude_auth({"CI": ci, **KEY}, signed_in=True) == sa.SUBSCRIPTION


def test_ci_never_probes_the_cli(monkeypatch):
    monkeypatch.setattr(sa, "claude_login_status",
                        lambda: pytest.fail("CI must not probe the Claude CLI"))
    assert sa.resolve_claude_auth({"CI": "true", **KEY}) == sa.KEY


def test_override_key_beats_a_login_and_skips_the_probe(monkeypatch):
    monkeypatch.setattr(sa, "claude_login_status",
                        lambda: pytest.fail("ROUTER_AUTH=key must not probe the Claude CLI"))
    assert sa.resolve_claude_auth({"ROUTER_AUTH": "key", **KEY}) == sa.KEY
    assert sa.resolve_claude_auth({"ROUTER_AUTH": "KEY"}) == sa.NONE


def test_override_subscription_never_falls_back_to_the_key():
    env = {"ROUTER_AUTH": "subscription", **KEY}
    assert sa.resolve_claude_auth(env, signed_in=True) == sa.SUBSCRIPTION
    assert sa.resolve_claude_auth(env, signed_in=False) == sa.NONE


def test_override_auto_and_unset_are_the_default_order():
    for env in ({"ROUTER_AUTH": "auto", **KEY}, {"ROUTER_AUTH": "", **KEY}, {**KEY}):
        assert sa.resolve_claude_auth(env, signed_in=True) == sa.SUBSCRIPTION
        assert sa.resolve_claude_auth(env, signed_in=False) == sa.KEY


def test_unknown_override_is_refused():
    with pytest.raises(ValueError, match="ROUTER_AUTH"):
        sa.resolve_claude_auth({"ROUTER_AUTH": "token"}, signed_in=True)


# --------------------------------------------------------------------------- #
# The CLI contract (subprocess faked)
# --------------------------------------------------------------------------- #
class FakeRun:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.calls = []
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


def test_login_probe_strips_keys_and_reads_logged_in(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-the-child")
    monkeypatch.setattr(sa.shutil, "which", lambda name: "/bin/claude")
    fake = FakeRun(json.dumps({"loggedIn": True, "authMethod": "claude.ai"}))
    monkeypatch.setattr(sa.subprocess, "run", fake)
    ok, detail = sa.claude_login_status()
    assert ok and "authMethod=claude.ai" in detail
    args, kwargs = fake.calls[0]
    assert args == ["/bin/claude", "auth", "status", "--json"]
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    sa.claude_login_status()
    assert len(fake.calls) == 1  # cached for the process


def test_login_probe_without_cli_or_with_bad_output_is_signed_out(monkeypatch):
    monkeypatch.setattr(sa.shutil, "which", lambda name: None)
    assert sa.claude_login_status()[0] is False
    monkeypatch.setattr(sa.shutil, "which", lambda name: "/bin/claude")
    monkeypatch.setattr(sa.subprocess, "run", FakeRun("not json"))
    assert sa.claude_login_status()[0] is False


def test_claude_print_runs_print_mode_with_tools_off_and_no_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-the-child")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-reach-the-child")
    monkeypatch.setattr(sa.shutil, "which", lambda name: "/bin/claude")
    fake = FakeRun(json.dumps({"type": "result", "is_error": False, "result": "ready",
                               "stop_reason": "end_turn",
                               "usage": {"input_tokens": 3, "output_tokens": 1}}))
    monkeypatch.setattr(sa.subprocess, "run", fake)
    data = REAL_CLAUDE_PRINT("say ready", "claude-sonnet-5", system="be brief", effort="high")
    args, kwargs = fake.calls[0]
    assert args[:2] == ["/bin/claude", "-p"]
    assert args[args.index("--model") + 1] == "claude-sonnet-5"
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--output-format") + 1] == "json"
    assert args[args.index("--system-prompt") + 1] == "be brief"
    assert args[args.index("--effort") + 1] == "high"
    assert "--bare" not in args  # bare mode never reads the subscription login
    assert "say ready" not in args and kwargs["input"] == "say ready"
    assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} & set(kwargs["env"])
    msg = sa.as_message(data)
    assert msg.content[0].text == "ready" and msg.usage.output_tokens == 1


@pytest.mark.parametrize("stdout,code", [
    (json.dumps({"is_error": True, "subtype": "error_during_execution"}), 1),
    (json.dumps({"is_error": True, "subtype": "error_max_turns"}), 0),
    ("", 1),
])
def test_claude_print_failures_raise(monkeypatch, stdout, code):
    monkeypatch.setattr(sa.shutil, "which", lambda name: "/bin/claude")
    monkeypatch.setattr(sa.subprocess, "run", FakeRun(stdout, code, "boom"))
    with pytest.raises(sa.ClaudeCLIError):
        REAL_CLAUDE_PRINT("x", "claude-haiku-4-5")


def test_codex_status_needs_the_chatgpt_login(monkeypatch):
    monkeypatch.setattr(sa.shutil, "which", lambda name: "/bin/codex")
    monkeypatch.setattr(sa.subprocess, "run", FakeRun("", 0, "Logged in using ChatGPT\n"))
    assert sa._codex_status()[0] is True
    monkeypatch.setattr(sa.subprocess, "run", FakeRun("", 0, "Logged in using an API key - sk-***\n"))
    assert sa._codex_status()[0] is False
    monkeypatch.setattr(sa.subprocess, "run", FakeRun("", 1, "Not logged in\n"))
    assert sa._codex_status()[0] is False


# --------------------------------------------------------------------------- #
# Router wiring
# --------------------------------------------------------------------------- #
def _router(monkeypatch, tmp_path, *, auth, key, signed_in, ci=None):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_ROUTER_OLLAMA_URL",
                "OLLAMA_API_KEY", "CI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROUTER_AUTH", auth)
    monkeypatch.setenv("CLAUDE_ROUTER_AUTO_ALLOCATE", "0")
    monkeypatch.setenv("CLAUDE_ROUTER_LOG", str(tmp_path / "usage.csv"))
    monkeypatch.setenv("CLAUDE_ROUTER_COMPACT", "0")
    if key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    if ci:
        monkeypatch.setenv("CI", ci)
    monkeypatch.setattr(sa, "claude_login_status", lambda: (signed_in, "faked"))
    import router
    return importlib.reload(router)


class _FakeClient:
    def __init__(self):
        self.calls = []
        client = self

        class Messages:
            def create(self, **kwargs):
                client.calls.append(kwargs)
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(text="from-key.")],
                    usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
                    stop_reason="end_turn")

        self.messages = Messages()


def test_router_uses_the_subscription_first_and_never_builds_a_client(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, auth="auto", key=True, signed_in=True)
    seen = []

    def fake_print(prompt, model, system=None, effort=None, json_schema=None, timeout=None):
        seen.append((prompt, model))
        return {"result": "from-subscription.", "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 3}}

    monkeypatch.setattr(sa, "claude_print", fake_print)
    monkeypatch.setattr(router, "_client", lambda: pytest.fail("API client used despite a login"))
    assert router.anthropic_ready()
    assert router.run("hello", tier="sonnet") == "from-subscription."
    assert seen == [("hello", "claude-sonnet-5")]
    assert "Claude Code CLI (subscription)" in (tmp_path / "usage.csv").read_text()


def test_router_subscription_works_without_the_sdk_or_a_key(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, auth="auto", key=False, signed_in=True)
    monkeypatch.setattr(router, "anthropic", None)
    monkeypatch.setattr(router, "_API_STATUS_ERRORS", ())
    monkeypatch.setattr(sa, "claude_print", lambda *a, **k: {"result": "ok.", "stop_reason": "end_turn"})
    assert router.anthropic_ready()
    assert router.run("hello", tier="haiku") == "ok."


def test_router_falls_back_to_the_key_when_claude_p_fails(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, auth="auto", key=True, signed_in=True)

    def broken(*a, **k):
        raise sa.ClaudeCLIError("claude -p exited 1")

    monkeypatch.setattr(sa, "claude_print", broken)
    client = _FakeClient()
    monkeypatch.setattr(router, "_client", lambda: client)
    assert router.run("hello", tier="sonnet") == "from-key."
    assert len(client.calls) == 1


def test_router_pinned_subscription_does_not_fall_back(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, auth="subscription", key=True, signed_in=True)

    def broken(*a, **k):
        raise sa.ClaudeCLIError("claude -p exited 1")

    monkeypatch.setattr(sa, "claude_print", broken)
    monkeypatch.setattr(router, "_client", lambda: pytest.fail("key used despite the pin"))
    with pytest.raises(router.RouterSetupError, match="no API-key fallback"):
        router.run("hello", tier="sonnet")


@pytest.mark.parametrize("auth,ci", [("key", None), ("auto", "true"), ("subscription", "true")])
def test_router_on_keys_under_ci_or_override(monkeypatch, tmp_path, auth, ci):
    router = _router(monkeypatch, tmp_path, auth=auth, key=True, signed_in=True, ci=ci)
    monkeypatch.setattr(sa, "claude_print", lambda *a, **k: pytest.fail("subscription used on keys"))
    client = _FakeClient()
    monkeypatch.setattr(router, "_client", lambda: client)
    assert router.run("hello", tier="sonnet") == "from-key."


def test_doctor_reports_auth_per_provider(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, auth="auto", key=False, signed_in=True)
    monkeypatch.setattr(sa, "_codex_status", lambda: (True, "codex login status"))
    monkeypatch.setattr(sa.shutil, "which", lambda name: None)
    monkeypatch.setenv("XAI_API_KEY", "x")
    monkeypatch.setattr(router, "_probe_base", lambda base, key: None)
    report = router.doctor()
    by = {row["provider"]: row["auth"] for row in report["auth"]}
    assert by == {"Claude": "subscription", "OpenAI": "subscription", "xAI Grok": "key",
                  "Gemini": "none", "Qwen": "none", "Ollama Cloud": "none"}
    assert report["claude_ready"] and report["claude_auth"] == "subscription"
    checks = {row["check"]: row for row in report["rows"]}
    assert checks["ANTHROPIC_API_KEY"]["ok"]  # a missing key is only the fallback here
    assert "auth: Claude" in checks and "auth: Ollama Cloud" in checks
    assert "test-key" not in json.dumps(report) and "x\"" not in json.dumps(report["auth"])


def test_doctor_under_forced_keys_probes_nothing(monkeypatch, tmp_path):
    router = _router(monkeypatch, tmp_path, auth="key", key=True, signed_in=True)
    monkeypatch.setattr(sa, "_codex_status", lambda: pytest.fail("probed under ROUTER_AUTH=key"))
    monkeypatch.setattr(sa, "claude_login_status", lambda: pytest.fail("probed under ROUTER_AUTH=key"))
    monkeypatch.setattr(router, "_probe_base", lambda base, key: None)
    report = router.doctor()
    by = {row["provider"]: row for row in report["auth"]}
    assert by["Claude"]["auth"] == "key" and by["OpenAI"]["auth"] in ("key", "none")
    assert "keys forced (ROUTER_AUTH=key)" in by["OpenAI"]["detail"]


# --------------------------------------------------------------------------- #
# hq-orchestrator worker caller
# --------------------------------------------------------------------------- #
@pytest.fixture
def server(monkeypatch):
    """Import hq_orchestrator.server with a stand-in FastMCP (no `mcp` needed)."""
    pytest.importorskip("anthropic")

    class FastMCP:
        def __init__(self, *_a, **_k):
            pass

        def tool(self):
            return lambda fn: fn

    fake = types.ModuleType("mcp.server.fastmcp")
    fake.FastMCP = FastMCP
    for name, mod in (("mcp", types.ModuleType("mcp")), ("mcp.server", types.ModuleType("mcp.server")),
                      ("mcp.server.fastmcp", fake)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "hq_orchestrator.server", raising=False)
    import hq_orchestrator.server as srv
    return srv


SUBMIT = {"name": "submit_result", "description": "d", "input_schema": {"type": "object"}}


def test_orchestrator_worker_uses_claude_p_structured_output(monkeypatch, server):
    monkeypatch.setenv("ROUTER_AUTH", "auto")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(sa, "claude_login_status", lambda: (True, "faked"))
    seen = {}

    def fake_print(prompt, model, system=None, effort=None, json_schema=None, timeout=None):
        seen.update(prompt=prompt, model=model, system=system, schema=json_schema)
        return {"structured_output": {"status": "completed"},
                "usage": {"input_tokens": 4, "output_tokens": 5}}

    monkeypatch.setattr(sa, "claude_print", fake_print)
    monkeypatch.setattr(server, "_client", None)  # any API use would crash
    result = server._anthropic_caller("claude-opus-5", "sys", "msg", SUBMIT)
    assert result["status"] == "completed" and "subscription" in result["usage_note"]
    assert seen == {"prompt": "msg", "model": "claude-opus-5", "system": "sys",
                    "schema": {"type": "object"}}


def test_orchestrator_worker_on_keys_in_ci(monkeypatch, server):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(sa, "claude_print", lambda *a, **k: pytest.fail("subscription used in CI"))
    block = types.SimpleNamespace(type="tool_use", name="submit_result", input={"status": "completed"})
    response = types.SimpleNamespace(content=[block],
                                     usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))
    monkeypatch.setattr(server, "_client", types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **k: response)))
    assert server._anthropic_caller("claude-opus-5", "s", "m", SUBMIT)["status"] == "completed"
