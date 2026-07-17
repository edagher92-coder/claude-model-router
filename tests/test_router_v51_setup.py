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
    # Isolate tests from the repo's real committed bench reports — allocation
    # tests opt back in explicitly with CLAUDE_ROUTER_AUTO_ALLOCATE=1.
    monkeypatch.setenv("CLAUDE_ROUTER_AUTO_ALLOCATE", "0")
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
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

    text = router.run("summarise this pile", tier="glm")
    assert text == LONG
    generate_calls = [u for u, _, _ in net.requests if u.endswith("/api/generate")]
    assert generate_calls == ["http://alive:11434/api/generate"]
    # Thinking must be disabled on every bridge dispatch (empty-response bug).
    gen_payload = [p for u, _, p in net.requests if u.endswith("/api/generate")][0]
    assert gen_payload["think"] is False
    # Cloud key rides along as a Bearer header on the daemon call too.
    auth = [h for u, h, _ in net.requests if u.endswith("/api/generate")][0]
    assert auth.get("Authorization") == "Bearer ok-123"


def test_generate_falls_back_to_cloud(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path,
                         ollama_url="http://dead:11434", ollama_key="ok-123")
    net = FakeOllamaNet({"https://ollama.com": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

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
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

    # No tier given: classification must not touch Anthropic in offline mode.
    assert router.run("draft a long research digest") == LONG
    assert not router.anthropic_ready()


def test_offline_claude_tier_reroutes_to_bridge(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

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
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

    # Incomplete at glm, nothing above is available offline -> honest return.
    assert router.run("hard task", tier="glm") == "too short"
    assert router._next_tier_up("glm", {"glm"}, stakes=False, claude_ok=False) is None


def test_offline_bridge_down_raises_setup_error(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://dead:11434")
    net = FakeOllamaNet({})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

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
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

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
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

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
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

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
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

    report = router.doctor()
    assert report["ok"] and not report["claude_ready"] and report["bridge_ready"]
    assert "OFFLINE" in report["mode"]


def test_doctor_flags_keyless_ollama_com(monkeypatch, tmp_path):
    """ollama.com answers /api/version without auth, but generation 401s —
    doctor must not report a keyless cloud base as a ready bridge."""
    router = load_router(monkeypatch, tmp_path, ollama_url="https://ollama.com")
    net = FakeOllamaNet({"https://ollama.com": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

    report = router.doctor()
    assert not report["bridge_ready"] and not report["ok"]
    row = next(r for r in report["rows"] if "ollama.com" in str(r["check"]))
    assert row["ok"] is False and "OLLAMA_API_KEY" in str(row["detail"])
    assert "OLLAMA_API_KEY" in str(row["fix"])


def test_doctor_unusable_gives_fixes_and_fails(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    net = FakeOllamaNet({})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

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

    monkeypatch.setattr(ollama_caller, "_dispatch_ssrf_ok", lambda base: None)
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

    monkeypatch.setattr(ollama_caller, "_dispatch_ssrf_ok", lambda base: None)
    monkeypatch.setattr(ollama_caller.urllib.request, "urlopen", fake_urlopen)
    result = ollama_caller.call("glm-5.2", "s", "m", {"input_schema": {"type": "object"}})
    assert result["status"] == "completed"
    assert calls == ["http://dead:11434/api/chat", "https://ollama.com/api/chat"]


# --------------------------------------------------------------------------- #
# Bench auto-allocation (added 2026-07-17)
# --------------------------------------------------------------------------- #
def _write_report(dir_, date, models):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{date}.json").write_text(json.dumps({"date": date, "models": models}))


def _probe_row(passes, latency=1.0, **extra):
    row = {p: {"pass": True, "latency_s": latency} for p in
           ("extract", "summarise", "code", "reason", "price-honesty", "tier-math")}
    for p in passes if isinstance(passes, list) else []:
        row[p]["pass"] = False
    row.update(extra)
    return row


def test_allocation_picks_fastest_clean_sweep(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_ROUTER_AUTO_ALLOCATE", "1")
    reports = tmp_path / "reports"
    _write_report(reports, "2026-07-17", {
        "glm-5.2": _probe_row([], latency=1.7),
        "kimi-k2.7-code": _probe_row([], latency=1.4),          # fastest sweep -> winner
        "nemotron-3-nano:30b": _probe_row(["tier-math"], 0.9),  # fails critical probe
        "claude-sonnet-5": _probe_row([], latency=0.5, baseline=True),  # never allocatable
    })
    monkeypatch.setattr(router, "BENCH_REPORTS_DIR", reports)

    alloc = router.bench_allocation()
    assert alloc["model"] == "kimi-k2.7-code"
    assert router._model_id("glm") == "kimi-k2.7-code"


def test_allocation_precedence_and_disable(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    reports = tmp_path / "reports"
    _write_report(reports, "2026-07-17", {"glm-5.2": _probe_row([], 1.0)})
    monkeypatch.setattr(router, "BENCH_REPORTS_DIR", reports)

    monkeypatch.setenv("CLAUDE_ROUTER_AUTO_ALLOCATE", "1")
    monkeypatch.setenv("GLM_OLLAMA_TAG", "pinned-tag")          # env beats bench
    assert router._model_id("glm") == "pinned-tag"
    monkeypatch.delenv("GLM_OLLAMA_TAG")
    monkeypatch.setenv("CLAUDE_ROUTER_AUTO_ALLOCATE", "0")      # kill switch
    assert router.bench_allocation() is None
    assert router._model_id("glm") == "glm-5.2:cloud"           # registry default


def test_allocation_none_without_report_or_qualifier(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_ROUTER_AUTO_ALLOCATE", "1")
    monkeypatch.setattr(router, "BENCH_REPORTS_DIR", tmp_path / "empty")
    assert router.bench_allocation() is None
    reports = tmp_path / "reports"
    # Report exists but its only model misses a critical probe entirely.
    row = {p: {"pass": True, "latency_s": 1.0} for p in ("extract", "reason")}
    _write_report(reports, "2026-07-17", {"some-model": row})
    monkeypatch.setattr(router, "BENCH_REPORTS_DIR", reports)
    assert router.bench_allocation() is None
    assert router._model_id("glm") == "glm-5.2:cloud"


# --------------------------------------------------------------------------- #
# Visibility — announce lines, via labels, --last (added 2026-07-17)
# --------------------------------------------------------------------------- #
def test_via_labels(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    assert "Anthropic API (ONLINE)" == router._via_label(None, "claude-sonnet-5")
    assert "Ollama Cloud (ONLINE)" == router._via_label("https://ollama.com", "glm-5.2")
    assert "local daemon (OFFLINE/on-prem)" == router._via_label("http://localhost:11434", "llama3.2:3b")
    lbl = router._via_label("http://100.122.28.89:11434", "glm-5.2:cloud")
    assert "routing server 100.122.28.89" in lbl and "compute on Ollama Cloud (ONLINE)" in lbl


def test_dispatch_announces_where_it_ran(monkeypatch, tmp_path, capsys):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

    router.run("bulk digest", tier="glm")
    err = capsys.readouterr().err
    assert "[router] glm -> glm-5.2:cloud via" in err
    # honest nuance: a :cloud tag through a private host still computes online
    assert "compute on Ollama Cloud (ONLINE)" in err

    # Silenceable for library use.
    monkeypatch.setenv("CLAUDE_ROUTER_ANNOUNCE", "0")
    router.run("bulk digest again", tier="glm")
    assert "[router]" not in capsys.readouterr().err


def test_usage_log_records_via_and_last_dispatches(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

    router.run("bulk digest", tier="glm")
    rows = router.last_dispatches(5)
    assert rows and rows[-1]["status"] == "ok"
    assert "compute on Ollama Cloud (ONLINE)" in rows[-1]["via"]


# --------------------------------------------------------------------------- #
# Status-line file for the desktop app (added 2026-07-17)
# --------------------------------------------------------------------------- #
def test_engine_of_maps_via_labels(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    assert router._engine_of("Anthropic API (ONLINE)") == "anthropic"
    assert router._engine_of("Ollama Cloud (ONLINE)") == "cloud"
    assert router._engine_of("routing server 100.x -> ':cloud' tag, compute on Ollama Cloud (ONLINE)") == "cloud"
    assert router._engine_of("local daemon (OFFLINE/on-prem)") == "local"


def test_dispatch_writes_routing_status(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="https://ollama.com", ollama_key="k")
    net = FakeOllamaNet({"https://ollama.com": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)
    status_path = tmp_path / ".routing-status.json"
    monkeypatch.setattr(router, "ROUTING_STATUS_FILE", status_path)

    router.run("bulk digest", tier="glm")
    state = json.loads(status_path.read_text())
    assert state["engine"] == "cloud" and state["tier"] == "glm" and "ts" in state

    monkeypatch.setenv("CLAUDE_ROUTER_STATUS", "0")   # kill switch
    status_path.unlink()
    router.run("again", tier="glm")
    assert not status_path.exists()


# --------------------------------------------------------------------------- #
# SEVERE #1 — stakes end-to-end (keyword backstop, not caller-optional)
# --------------------------------------------------------------------------- #
def test_looks_like_stakes_is_precise_and_lenient(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    # customer money/legal -> stakes
    for s in ["draft the invoice for the refund", "process the $500 chargeback",
              "the GST tax invoice", "send a payment link", "review the contract terms and conditions"]:
        assert router.looks_like_stakes(s), s
    # engineering / analysis -> NOT stakes (lenient: runs on the bridge)
    for s in ["run a top-to-bottom valuation and review of the router",
              "audit the codebase", "refactor the pricing module", "evaluate the models",
              "summarise this research corpus", "review this pull request"]:
        assert not router.looks_like_stakes(s), s


class _FakeClaude:
    def __init__(self, text="CLAUDE ANSWER, complete and long enough to pass"):
        self._text = text
        class _M:
            def create(_s, **kw):
                self.seen_model = kw["model"]
                return type("R", (), {
                    "content": [type("B", (), {"text": self._text})()],
                    "usage": type("U", (), {"input_tokens": 1, "output_tokens": 3})(),
                    "stop_reason": "end_turn"})()
        self.messages = _M()


def test_stakes_keyword_forces_claude_and_never_hits_bridge(monkeypatch, tmp_path, capsys):
    router = load_router(monkeypatch, tmp_path, anthropic_key="test-key")
    fake = _FakeClaude()
    monkeypatch.setattr(router, "_client", lambda: fake)
    # Hard proof: the bridge path must never be entered for a stakes task.
    def _boom(*a, **k):
        raise AssertionError("stakes task reached the Ollama bridge!")
    monkeypatch.setattr(router, "_ollama_generate", _boom)

    # Caller FORGOT stakes=True and even asked for the glm tier explicitly.
    out = router.run("draft the invoice email for the Merivale refund", tier="glm")
    assert out == "CLAUDE ANSWER, complete and long enough to pass"
    assert fake.seen_model == "claude-sonnet-5"   # glm -> rerouted to Claude
    assert "NUMBERS RULE" in capsys.readouterr().err


def test_stakes_keyword_offline_refuses(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")  # no claude
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)
    try:
        router.run("send the customer their $450 tax invoice", tier="glm")
    except router.RouterSetupError as exc:
        assert "NUMBERS RULE" in str(exc)
    else:
        raise AssertionError("stakes task offline must refuse, not hit the bridge")


def test_lenient_review_task_runs_on_bridge_offline(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, ollama_url="http://alive:11434")
    net = FakeOllamaNet({"http://alive:11434": LONG})
    monkeypatch.setattr(router.urllib.request, "urlopen", net)
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)
    # A model valuation / code review is NOT stakes — strongest bridge model runs it.
    assert router.run("run a top-to-bottom valuation and review of the router") == LONG


# --------------------------------------------------------------------------- #
# SEVERE #2 — SSRF allowlist on the Ollama base URL
# --------------------------------------------------------------------------- #
def test_ssrf_allow_and_block_matrix(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    ok = ["http://localhost:11434", "http://127.0.0.1:11434", "https://ollama.com",
          "http://100.122.28.89:11434", "http://192.168.1.10:11434", "http://10.0.0.5:11434",
          "http://elzydlab.tail76b098.ts.net:11434"]
    bad = ["http://169.254.169.254/latest/meta-data/", "file:///etc/passwd",
           "http://attacker.com/steal", "https://8.8.8.8", "ftp://localhost/x",
           "gopher://127.0.0.1"]
    for u in ok:
        assert router.is_allowed_base(u)[0], u
    for u in bad:
        assert not router.is_allowed_base(u)[0], u


def test_ssrf_explicit_allowlist_opens_a_public_host(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    assert not router.is_allowed_base("http://my-gpu-box.example.com:11434")[0]
    monkeypatch.setenv("CLAUDE_ROUTER_OLLAMA_ALLOW_HOSTS", "my-gpu-box.example.com")
    assert router.is_allowed_base("http://my-gpu-box.example.com:11434")[0]


def test_ssrf_blocked_base_is_dropped_from_chain(monkeypatch, tmp_path, capsys):
    router = load_router(monkeypatch, tmp_path,
                         ollama_url="http://169.254.169.254,http://100.122.28.89:11434",
                         ollama_key="k")
    bases = [b for b, _ in router._ollama_bases()]
    assert "http://169.254.169.254" not in bases
    assert "http://100.122.28.89:11434" in bases
    assert "BLOCKED" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Kimi 3rd-pass adjudicated fixes (2026-07-17)
# --------------------------------------------------------------------------- #
def test_dns_resolution_guard_blocks_rebind(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    # a single-label / private-looking name that actually resolves PUBLIC
    monkeypatch.setattr(router.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 0))])
    assert router._host_resolves_private("sneaky") is False
    import pytest
    with pytest.raises(RuntimeError, match="SSRF guard"):
        router._dispatch_ssrf_ok("http://sneaky:11434")
    # resolves to a private IP -> allowed
    monkeypatch.setattr(router.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("10.1.2.3", 0))])
    assert router._host_resolves_private("gpu-box") is True
    router._dispatch_ssrf_ok("http://gpu-box:11434")   # no raise
    # metadata IP among results -> rejected
    monkeypatch.setattr(router.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("169.254.169.254", 0))])
    assert router._host_resolves_private("evil") is False
    # ollama.com is an intentional public target -> dispatch guard skips it
    router._dispatch_ssrf_ok("https://ollama.com")


def test_allowlist_entry_with_port_matches(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    monkeypatch.setenv("CLAUDE_ROUTER_OLLAMA_ALLOW_HOSTS", "my-gpu.example.com:11434")
    assert router.is_allowed_base("http://my-gpu.example.com:11434")[0]


def test_done_reason_length_escalates(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, anthropic_key="k", ollama_url="http://alive:11434")
    monkeypatch.setattr(router, "_host_resolves_private", lambda h: True)

    class Net:
        def __call__(self, request, timeout=None):
            url = request.full_url
            if url.endswith("/api/version"):
                return _resp({"version": "1"})
            if url.endswith("/api/generate"):
                # long text BUT truncated by the token cap
                return _resp({"response": "x" * 80, "done_reason": "length",
                              "prompt_eval_count": 1, "eval_count": 1})
            return _resp({})
    monkeypatch.setattr(router.urllib.request, "urlopen", Net())
    fake = _FakeClaude("A full Claude answer after the truncated bridge reply.")
    monkeypatch.setattr(router, "_client", lambda: fake)
    # glm truncated -> one-strike escalates up to a Claude tier
    out = router.run("bulk", tier="glm")
    assert out == "A full Claude answer after the truncated bridge reply."


def test_log_write_never_crashes_dispatch(monkeypatch, tmp_path, capsys):
    router = load_router(monkeypatch, tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a dir")
    monkeypatch.setattr(router, "_log_path", lambda: blocker / "sub" / "write.csv")
    router._log("glm", "glm-5.2", "", 1, 2, "ok", via="test")   # must not raise
    assert "usage-log write failed" in capsys.readouterr().err


def test_csv_formula_injection_neutralised(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path)
    assert router._csv_safe("=cmd|' /c calc'!A1") == "'=cmd|' /c calc'!A1"
    assert router._csv_safe("+1+1") == "'+1+1"
    assert router._csv_safe("glm-5.2") == "glm-5.2"   # normal value untouched


def test_output_config_fallback_on_typeerror(monkeypatch, tmp_path):
    router = load_router(monkeypatch, tmp_path, anthropic_key="k")
    calls = []

    class Msgs:
        def create(self, **kw):
            calls.append(kw)
            if "output_config" in kw:
                raise TypeError("create() got an unexpected keyword argument 'output_config'")
            return type("R", (), {"content": [type("B", (), {"text": "ok answer, long enough here"})()],
                                  "usage": type("U", (), {"input_tokens": 1, "output_tokens": 2})(),
                                  "stop_reason": "end_turn"})()

    class Client:
        messages = Msgs()
    monkeypatch.setattr(router, "_client", lambda: Client())
    out = router.run("normal task", tier="sonnet", effort="high")
    assert out == "ok answer, long enough here"
    assert len(calls) == 2 and "output_config" not in calls[1]   # retried without it
