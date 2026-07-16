"""v5.1 setup contract — offline (Ollama-only) mode, the base chain
(routing server -> second host -> Ollama Cloud), doctor, and call-time env.

Everything runs with zero network and zero real keys: urllib is faked and
the Anthropic client is never constructed.
"""
import importlib
import io
import json
import urllib.error


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def load_router(monkeypatch, tmp_path, *, anthropic_key=None, ollama_url=None,
                ollama_key=None):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_ROUTER_OLLAMA_URL", "OLLAMA_API_KEY",
                "GLM_OLLAMA_TAG", "CLAUDE_ROUTER_GLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    if anthropic_key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    if ollama_url:
        monkeypatch.setenv("CLAUDE_ROUTER_OLLAMA_URL", ollama_url)
    if ollama_key:
        monkeypatch.setenv("OLLAMA_API_KEY", ollama_key)
    monkeypatch.setenv("CLAUDE_ROUTER_LOG", str(tmp_path / "usage.csv"))
    import router

    return importlib.reload(router)


class FakeOllamaNet:
    """urllib.request.urlopen stand-in: a map of reachable bases and canned
    /api/generate responses; unreachable bases raise URLError."""

    def __init__(self, up: dict, version: str = "0.9.9"):
        # up: base_url -> reply text for /api/generate (or None: up but errors)
        self.up = up
        self.version = version
        self.requests = []  # (url, headers, payload|None)

    def __call__(self, request, timeout=None):
        url = request.full_url
        payload = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append((url, dict(request.header_items()), payload))
        base = url.rsplit("/api/", 1)[0]
        if base not in self.up:
            raise urllib.error.URLError("unreachable")
        if url.endswith("/api/version"):
            return _resp({"version": self.version})
        if url.endswith("/api/tags"):
            return _resp({"models": [{"name": "glm-5.2:cloud"}]})
        if url.endswith("/api/generate"):
            reply = self.up[base]
            if reply is None:
                return _resp({"error": "boom", "response": ""})
            return _resp({"response": reply, "prompt_eval_count": 7, "eval_count": 11})
        raise AssertionError(f"unexpected URL {url}")


def _resp(body: dict):
    class _Ctx:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self._data).encode("utf-8")

    return _Ctx(body)


LONG = "A thorough, complete and confident answer to the delegated bulk task."


# --------------------------------------------------------------------------- #
# Base chain resolution
# --------------------------------------------------------------------------- #
def test_default_chain_is_localhost_only(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    assert router._ollama_bases() == [("http://localhost:11434", "")]


def test_cloud_key_appends_ollama_com(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_key="ok-123")
    assert router._ollama_bases() == [
        ("http://localhost:11434", "ok-123"),
        ("https://ollama.com", "ok-123"),
    ]


def test_comma_separated_tailnet_chain(monkeypatch, tmp_path):
    router = load_router(
        monkeypatch, tmp_path,
        ollama_url="http://100.122.28.89:11434, http://100.81.52.33:11434",
        ollama_key="ok-123",
    )
    assert [b for b, _ in router._ollama_bases()] == [
        "http://100.122.28.89:11434",
        "http://100.81.52.33:11434",
        "https://ollama.com",
    ]


def test_explicit_cloud_url_not_duplicated(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path,
                         ollama_url="https://ollama.com", ollama_key="ok-123")
    assert router._ollama_bases() == [("https://ollama.com", "ok-123")]


def test_glm_tag_precedence(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    assert router._model_id("glm") == "glm-5.2:cloud"
    monkeypatch.setenv("GLM_OLLAMA_TAG", "glm-5.2:local")
    assert router._model_id("glm") == "glm-5.2:local"
    monkeypatch.setenv("CLAUDE_ROUTER_GLM_MODEL", "custom-glm")
    assert router._model_id("glm") == "custom-glm"


# --------------------------------------------------------------------------- #
# Chain failover at dispatch time
# --------------------------------------------------------------------------- #
def test_generate_uses_first_reachable_base(monkeypatch, tmp_path):
    router = load_router(
        monkeypatch, tmp_path,
        ollama_url="http://dead:11434,http://alive:11434", ollama_key="ok-123",
    )
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    text = router.run("summarise this pile", tier="glm")
    assert text == LONG
    generate_calls = [u for u, _, _ in net.requests if u.endswith("/api/generate")]
    assert generate_calls == ["http://alive:11434/api/generate"]
    # Cloud key rides along as a Bearer header on the daemon call too.
    auth = [h for u, h, _ in net.requests if u.endswith("/api/generate")][0]
    assert auth.get("Authorization") == "Bearer ok-123"


def test_generate_falls_back_to_cloud(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path,
                         ollama_url="http://dead:11434", ollama_key="ok-123")
    net = FakeOllamaNet({"https://ollama.com": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    assert router.run("bulk digest", tier="glm") == LONG
    generate_calls = [u for u, _, _ in net.requests if u.endswith("/api/generate")]
    assert generate_calls == ["https://ollama.com/api/generate"]


# --------------------------------------------------------------------------- #
# OFFLINE mode (no Anthropic engine)
# --------------------------------------------------------------------------- #
def test_offline_run_lands_on_bridge_without_classifier(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    # No tier given: classification must not touch Anthropic in offline mode.
    assert router.run("draft a long research digest") == LONG
    assert not router.anthropic_ready()


def test_offline_claude_tier_reroutes_to_bridge(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    assert router.run("summarise", tier="opus") == LONG  # rerouted, not crashed


def test_offline_stakes_refuses_cleanly(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    try:
        router.run("quote the customer $500", tier="sonnet", stakes=True)
    except router.RouterSetupError as exc:
        assert "NUMBERS RULE" in str(exc)
    else:
        raise AssertionError("stakes offline must raise RouterSetupError")


def test_offline_escalation_never_climbs_into_claude(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": "too short"})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    # Incomplete at glm, nothing above is available offline -> honest return.
    assert router.run("hard task", tier="glm") == "too short"
    assert router._next_tier_up("glm", {"glm"}, stakes=False, claude_ok=False) is None


def test_offline_bridge_down_raises_setup_error(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://dead:11434")
    net = FakeOllamaNet({})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    try:
        router.run("anything", tier="glm")
    except router.RouterSetupError as exc:
        assert "--doctor" in str(exc)
    else:
        raise AssertionError("no engine at all must raise RouterSetupError")


def test_offline_no_bridge_configured_raises(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)  # no URL, no keys
    net = FakeOllamaNet({})  # localhost probe fails too
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    try:
        router.run("anything", tier="sonnet")
    except router.RouterSetupError as exc:
        assert "no engine available" in str(exc)
    else:
        raise AssertionError("expected RouterSetupError")


def test_online_bridge_failure_still_recovers_on_sonnet(monkeypatch, tmp_path):
    """Regression: with Anthropic available, a dead bridge falls back to
    sonnet exactly as v5.1 shipped."""
    router = load_router(monkeypatch, tmp_path, anthropic_key="test-key",
                         ollama_url="http://dead:11434")
    net = FakeOllamaNet({})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    class FakeResponse:
        content = [type("B", (), {"text": LONG})()]
        usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()
        stop_reason = "end_turn"

    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["model"] == "claude-sonnet-5"
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(router, "_client", lambda: FakeClient())
    assert router.run("bulk digest", tier="glm") == LONG


# --------------------------------------------------------------------------- #
# Doctor
# --------------------------------------------------------------------------- #
def test_doctor_full_ladder(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, anthropic_key="test-key",
                         ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    report = router.doctor()
    assert report["ok"] and report["claude_ready"] and report["bridge_ready"]
    assert report["bridge_base"] == "http://alive:11434"
    assert "full ladder" in report["mode"]
    checks = {row["check"] for row in report["rows"]}
    assert "ANTHROPIC_API_KEY" in checks and "GLM model tag" in checks


def test_doctor_offline_mode(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    report = router.doctor()
    assert report["ok"] and not report["claude_ready"] and report["bridge_ready"]
    assert "OFFLINE" in report["mode"]


def test_doctor_flags_keyless_ollama_com(monkeypatch, tmp_path):
    """ollama.com answers /api/version without auth, but generation 401s —
    doctor must not report a keyless cloud base as a ready bridge."""
    router = load_router(monkeypatch, tmp_path, ollama_url="https://ollama.com")
    net = FakeOllamaNet({"https://ollama.com": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    report = router.doctor()
    assert not report["bridge_ready"] and not report["ok"]
    row = next(r for r in report["rows"] if "ollama.com" in str(r["check"]))
    assert row["ok"] is False and "OLLAMA_API_KEY" in str(row["detail"])
    assert "OLLAMA_API_KEY" in str(row["fix"])


def test_doctor_unusable_gives_fixes_and_fails(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    net = FakeOllamaNet({})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)

    report = router.doctor()
    assert not report["ok"] and "unusable" in report["mode"]
    fixes = " ".join(str(row["fix"]) for row in report["rows"])
    assert "ANTHROPIC_API_KEY" in fixes and "CLAUDE_ROUTER_OLLAMA_URL" in fixes
    # Doctor must never print or return secrets — only set/unset.
    assert "test-key" not in json.dumps(report)


# --------------------------------------------------------------------------- #
# ollama_caller: env is read at CALL time (no import-time freeze)
# --------------------------------------------------------------------------- #
def test_ollama_caller_reads_env_at_call_time(monkeypatch, tmp_path):
    from hq_orchestrator import ollama_caller

    # Set env AFTER import — the old module froze these at import time.
    monkeypatch.setenv("CLAUDE_ROUTER_OLLAMA_URL", "http://alive:11434")
    monkeypatch.setenv("GLM_OLLAMA_TAG", "glm-5.2:local")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    envelope = {"envelope_version": "1.0", "run_id": "r", "task_id": "T001",
                "status": "completed", "self_check": {"verified": [], "unverified": []}}
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return _resp({"message": {"content": json.dumps(envelope)},
                      "prompt_eval_count": 1, "eval_count": 2})

    monkeypatch.setattr(ollama_caller.urllib.request, "urlopen", fake_urlopen)
    result = ollama_caller.call("glm-5.2", "system", "message",
                                {"input_schema": {"type": "object"}})
    assert seen["url"] == "http://alive:11434/api/chat"
    assert seen["payload"]["model"] == "glm-5.2:local"
    assert result["status"] == "completed"
    assert "glm-5.2:local@http://alive:11434" in result["usage_note"]


def test_ollama_caller_chain_falls_back_to_cloud(monkeypatch, tmp_path):
    from hq_orchestrator import ollama_caller

    monkeypatch.setenv("CLAUDE_ROUTER_OLLAMA_URL", "http://dead:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "ok-123")
    monkeypatch.setattr(ollama_caller, "RETRY_DELAYS", ())  # no sleeps in tests

    envelope = {"envelope_version": "1.0", "run_id": "r", "task_id": "T001",
                "status": "completed", "self_check": {"verified": [], "unverified": []}}
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        if request.full_url.startswith("http://dead"):
            raise urllib.error.URLError("unreachable")
        assert dict(request.header_items()).get("Authorization") == "Bearer ok-123"
        return _resp({"message": {"content": json.dumps(envelope)}})

    monkeypatch.setattr(ollama_caller.urllib.request, "urlopen", fake_urlopen)
    result = ollama_caller.call("glm-5.2", "s", "m", {"input_schema": {"type": "object"}})
    assert result["status"] == "completed"
    assert calls == ["http://dead:11434/api/chat", "https://ollama.com/api/chat"]
