"""NUMBERS RULE + SSRF-redirect hardening — the critical invariants for the
direct (non-router) dispatch paths. Zero network, zero real keys.

Covered:
(a) the ollama_route CLI refuses a customer money/legal prompt BEFORE any dispatch;
(b) a direct ollama_caller.call() refuses stakes content the same way;
(c) an HTTP redirect (e.g. 302 -> 169.254.169.254) is rejected, never followed;
(d) ollama_route re-validates the base at DISPATCH time (DNS-rebinding / TOCTOU),
    not only at import.
"""
import importlib
import importlib.util
import io
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_STAKES_ENV = (
    "CLAUDE_ROUTER_OLLAMA_URL", "OLLAMA_API_KEY",
    "CLAUDE_ROUTER_OLLAMA_ALLOW_HOSTS", "CLAUDE_ROUTER_STAKES_KEYWORDS",
)


def _load_ollama_route(monkeypatch, *, ollama_url=None, ollama_key=None):
    """Import .claude/tools/ollama_route.py fresh under a controlled env.
    (It lives outside any package, so load it by path.)"""
    for var in _STAKES_ENV:
        monkeypatch.delenv(var, raising=False)
    if ollama_url:
        monkeypatch.setenv("CLAUDE_ROUTER_OLLAMA_URL", ollama_url)
    if ollama_key:
        monkeypatch.setenv("OLLAMA_API_KEY", ollama_key)
    path = ROOT / ".claude" / "tools" / "ollama_route.py"
    spec = importlib.util.spec_from_file_location("ollama_route_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fresh_caller(monkeypatch):
    for var in _STAKES_ENV:
        monkeypatch.delenv(var, raising=False)
    from hq_orchestrator import ollama_caller
    return importlib.reload(ollama_caller)


# --------------------------------------------------------------------------- #
# (a) CLI refuses a stakes prompt before dispatch
# --------------------------------------------------------------------------- #
def test_ollama_route_cli_refuses_stakes_invoice(monkeypatch, capsys):
    mod = _load_ollama_route(monkeypatch, ollama_key="k")

    def _boom(*a, **k):
        raise AssertionError("dispatch attempted on a stakes prompt")

    monkeypatch.setattr(mod, "generate", _boom)
    monkeypatch.setattr(
        sys, "argv",
        ["ollama_route", "--route", "heavy-reason",
         "Draft a tax invoice for $450 owed by the customer"],
    )
    rc = mod.main()
    assert rc == 4
    assert "NUMBERS RULE" in capsys.readouterr().err


def test_ollama_route_cli_allows_plain_prompt(monkeypatch, capsys):
    mod = _load_ollama_route(monkeypatch, ollama_key="k")
    seen = {}

    def _fake_generate(model, prompt, system, timeout):
        seen["model"] = model
        return {"response": "refactored ok"}

    monkeypatch.setattr(mod, "generate", _fake_generate)
    monkeypatch.setattr(mod, "_write_status", lambda *a, **k: None)
    monkeypatch.setattr(
        sys, "argv",
        ["ollama_route", "--route", "heavy-code", "Refactor this parser for clarity"],
    )
    rc = mod.main()
    assert rc == 0
    assert seen["model"] == "kimi-k2.7-code:cloud"
    assert "refactored ok" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# (b) direct ollama_caller.call() refuses stakes content
# --------------------------------------------------------------------------- #
def test_ollama_caller_call_refuses_stakes(monkeypatch):
    caller = _fresh_caller(monkeypatch)

    def _no_net(*a, **k):
        raise AssertionError("network reached on a stakes call")

    monkeypatch.setattr(caller.urllib.request, "urlopen", _no_net)
    with pytest.raises(RuntimeError, match="NUMBERS RULE"):
        caller.call(
            "glm-5.2",
            system="You are a helpful assistant.",
            message="Issue a refund of $20 to the customer's credit card please.",
            submit_tool={"input_schema": {"type": "object"}},
        )


def test_ollama_caller_stakes_scans_system_prompt_too(monkeypatch):
    caller = _fresh_caller(monkeypatch)
    monkeypatch.setattr(
        caller.urllib.request, "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network reached")),
    )
    with pytest.raises(RuntimeError, match="NUMBERS RULE"):
        caller.call(
            "glm-5.2",
            system="You draft customer tax invoice documents.",
            message="Summarise the attached notes.",
            submit_tool={"input_schema": {"type": "object"}},
        )


# --------------------------------------------------------------------------- #
# (c) HTTP redirects are refused, never followed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("modname", ["router", "hq_orchestrator.ollama_caller"])
def test_redirect_to_metadata_is_refused(modname):
    import urllib.error
    import urllib.request
    mod = importlib.import_module(modname)
    handler = mod._NoRedirect()
    req = urllib.request.Request("http://ollama.com/api/chat")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        handler.redirect_request(
            req, io.BytesIO(b""), 302, "Found", {},
            "http://169.254.169.254/latest/meta-data/",
        )
    assert excinfo.value.code == 302
    assert "blocked" in str(excinfo.value)


def test_global_opener_has_no_redirect_handler():
    """Importing the router installs a process-wide no-redirect opener, so even a
    urllib.request.urlopen elsewhere (e.g. the webhook path) fails closed on 30x."""
    import router  # noqa: F401  (import for its install_opener side effect)
    import urllib.request
    opener = urllib.request._opener
    assert opener is not None
    assert any(
        isinstance(h, urllib.request.HTTPRedirectHandler)
        and type(h) is not urllib.request.HTTPRedirectHandler
        for h in opener.handlers
    ), "no custom (redirect-blocking) HTTPRedirectHandler is installed"


# --------------------------------------------------------------------------- #
# (d) ollama_route re-validates the base at dispatch time (rebinding / TOCTOU)
# --------------------------------------------------------------------------- #
def test_ollama_route_dispatch_reresolves_host(monkeypatch):
    # "myhub" is a single-label LAN name: allowed by the static import-time check,
    # but at dispatch its live DNS answer decides. Simulate it flipping to public.
    mod = _load_ollama_route(monkeypatch, ollama_url="http://myhub")
    monkeypatch.setattr(mod, "_resolves_private", lambda host: False)
    with pytest.raises(RuntimeError, match="SSRF guard"):
        mod._get("/api/tags", 5)


def test_ollama_route_dispatch_allows_ollama_com(monkeypatch):
    mod = _load_ollama_route(monkeypatch, ollama_key="k")  # BASE=https://ollama.com
    mod._dispatch_ssrf_ok()  # intentional public target — must not raise


def test_ollama_route_dispatch_blocks_public_ip(monkeypatch):
    # A literal public IP must never survive to dispatch, even if env is tampered
    # after import.
    mod = _load_ollama_route(monkeypatch, ollama_key="k")
    monkeypatch.setattr(mod, "BASE", "http://8.8.8.8:11434")
    with pytest.raises(RuntimeError, match="SSRF|blocked"):
        mod._dispatch_ssrf_ok()
