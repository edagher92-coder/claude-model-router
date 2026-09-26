"""Hosted OpenAI-compatible bench providers (Qwen, Gemini, GLM/Zhipu, Grok).

Everything runs offline: urllib.request.urlopen is faked (no real HTTP), no
key ever leaves this process, and no key is ever printed. These tests pin the
contract the task description asked for:

  - each provider is controlled purely by its own key/base/models env vars
    and --<flag>-models/--<flag>-list CLI flags — no model id is hardcoded;
  - a provider with no key is skipped cleanly (a printed note, exit 0),
    never an exception that kills the whole bench;
  - --<provider>-list uses the provider's own /models endpoint, never a
    guessed id, and refuses cleanly (exit 1) when the key is missing;
  - every bench row (any provider) carries latency_s and tokens next to the
    pass/fail score, so speed/volume can be read alongside capability.
"""
import importlib
import io
import json
import pathlib
import sys
import urllib.error

import pytest

BENCH_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "bench")
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)
import model_bench  # noqa: E402


def reload_bench():
    return importlib.reload(model_bench)


class FakeCompatNet:
    """urllib.request.urlopen stand-in for the OpenAI-compatible adapter.
    up: base_url -> {"chat": reply_text, "models": [ids]}. Records every
    request so a test can assert exactly what was (and wasn't) called, and
    that no Authorization header ever leaks a real-looking secret value."""

    def __init__(self, up: dict):
        self.up = up
        self.requests = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        headers = dict(request.header_items())
        payload = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append((url, headers, payload))
        for base, spec in self.up.items():
            if not url.startswith(base):
                continue
            if url.endswith("/chat/completions"):
                return _resp({
                    "choices": [{"message": {"content": spec["chat"]}}],
                    "usage": {"completion_tokens": spec.get("tokens", 7)},
                })
            if url.endswith("/models"):
                return _resp({"data": [{"id": m} for m in spec.get("models", [])]})
        raise urllib.error.URLError(f"unreachable: {url}")


def _resp(body: dict):
    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(body).encode("utf-8")

    return _Ctx()


# --------------------------------------------------------------------------- #
# Provider registry shape
# --------------------------------------------------------------------------- #
def test_all_four_providers_registered_with_distinct_env_and_flags():
    bench = reload_bench()
    by_key = {p.key: p for p in bench.COMPAT_PROVIDERS}

    assert set(by_key) == {"qwen", "gemini", "glm", "grok"}
    assert by_key["qwen"].key_env == "QWEN_API_KEY"
    assert by_key["qwen"].base_env == "QWEN_BASE_URL"
    assert by_key["qwen"].models_env == "QWEN_BENCH_MODELS"
    assert by_key["qwen"].flag == "qwen"

    assert by_key["gemini"].key_env == "GEMINI_API_KEY"
    assert by_key["gemini"].base_env == "GEMINI_BASE_URL"
    assert by_key["gemini"].models_env == "GEMINI_BENCH_MODELS"
    assert by_key["gemini"].flag == "gemini"

    assert by_key["glm"].key_env == "GLM_API_KEY"
    assert by_key["glm"].base_env == "GLM_BASE_URL"
    assert by_key["glm"].models_env == "GLM_BENCH_MODELS"
    assert by_key["glm"].flag == "glm"

    # Grok's env vars are XAI_*, but its CLI flags/label say "grok" — exactly
    # as the task asked for (env prefix and flag prefix differ on purpose).
    assert by_key["grok"].key_env == "XAI_API_KEY"
    assert by_key["grok"].base_env == "XAI_BASE_URL"
    assert by_key["grok"].models_env == "XAI_BENCH_MODELS"
    assert by_key["grok"].flag == "grok"

    # No env var or default base is shared between providers.
    assert len({p.key_env for p in bench.COMPAT_PROVIDERS}) == 4
    assert len({p.default_base for p in bench.COMPAT_PROVIDERS}) == 4


def test_no_model_ids_are_hardcoded_anywhere_in_the_module():
    import pathlib

    module_path = pathlib.Path(__file__).resolve().parent.parent / "bench" / "model_bench.py"
    source = module_path.read_text(encoding="utf-8")
    # Nothing that looks like a Gemini/GLM/Grok model id literal (the task's
    # own example "Gemini 3.8 Flash") should appear in source.
    for banned in ("gemini-3.8-flash", "glm-4.6", "grok-4"):
        assert banned not in source.lower()


# --------------------------------------------------------------------------- #
# generate_openai_compat / discover_openai_compat (provider-agnostic adapter)
# --------------------------------------------------------------------------- #
def test_generate_openai_compat_records_latency_and_tokens(monkeypatch):
    bench = reload_bench()
    net = FakeCompatNet({"https://example.test/v1": {"chat": "Carol", "tokens": 42}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)

    text, latency, tokens = bench.generate_openai_compat(
        "https://example.test/v1", "sk-fake-not-real", "some-model", "who is shortest?")

    assert text == "Carol"
    assert tokens == 42
    assert latency >= 0
    # The fake key must reach the Authorization header (that's how the real
    # API would authenticate) but never anywhere else observable in the test.
    url, headers, payload = net.requests[0]
    assert headers["Authorization"] == "Bearer sk-fake-not-real"
    assert payload["model"] == "some-model"


def test_generate_openai_compat_retries_without_thinking_flag_on_400(monkeypatch):
    bench = reload_bench()
    calls = []

    def flaky(request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(payload)
        if "enable_thinking" in payload:
            raise urllib.error.HTTPError(request.full_url, 400, "bad request", {}, io.BytesIO(b"{}"))
        return _resp({"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 3}})

    monkeypatch.setattr(bench.urllib.request, "urlopen", flaky)
    text, _, tokens = bench.generate_openai_compat("https://example.test/v1", "k", "m", "p")

    assert text == "ok"
    assert tokens == 3
    assert len(calls) == 2 and "enable_thinking" not in calls[1]


def test_discover_openai_compat_lists_ids_and_never_guesses(monkeypatch):
    bench = reload_bench()
    net = FakeCompatNet({
        "https://example.test/v1": {"chat": "", "models": ["gemini-3.8-flash-preview", "gemini-2.5-pro"]},
    })
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)

    ids = bench.discover_openai_compat("https://example.test/v1", "k", label="gemini")

    assert ids == ["gemini-3.8-flash-preview", "gemini-2.5-pro"]


def test_discover_openai_compat_failure_is_best_effort(monkeypatch, capsys):
    bench = reload_bench()

    def boom(request, timeout=None):
        raise urllib.error.URLError("dns fail")

    monkeypatch.setattr(bench.urllib.request, "urlopen", boom)
    ids = bench.discover_openai_compat("https://example.test/v1", "k", label="glm")

    assert ids == []
    assert "glm discover" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# main(): clean skip without a key, never an exception
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Belt-and-braces: any urlopen call not explicitly faked by a test fails
    loudly instead of silently reaching the real internet."""
    def refuse(*a, **k):
        raise AssertionError("real network call attempted in a test")

    monkeypatch.setattr("urllib.request.urlopen", refuse)


def _clear_provider_env(monkeypatch, bench):
    for p in bench.COMPAT_PROVIDERS:
        monkeypatch.delenv(p.key_env, raising=False)
        monkeypatch.delenv(p.base_env, raising=False)
        monkeypatch.delenv(p.models_env, raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_each_provider_skips_cleanly_without_its_key(monkeypatch, tmp_path, capsys):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    argv = [
        "model_bench.py", "--models", "", "--baselines", "",
        "--out-dir", str(tmp_path),
        "--qwen-models", "qwen-fake", "--gemini-models", "gemini-fake",
        "--glm-models", "glm-fake", "--grok-models", "grok-fake",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = bench.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "QWEN_API_KEY unset" in out
    assert "GEMINI_API_KEY unset" in out
    assert "GLM_API_KEY unset" in out
    assert "XAI_API_KEY unset" in out
    report = json.loads((tmp_path / f"{bench.dt.date.today().isoformat()}.json").read_text())
    assert report["models"] == {}


def test_provider_with_key_is_benched_and_labelled(monkeypatch, tmp_path):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-not-real")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.test/gemini")

    net = FakeCompatNet({"https://example.test/gemini": {"chat": "Carol 2 5", "tokens": 5}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)

    argv = [
        "model_bench.py", "--models", "", "--baselines", "",
        "--out-dir", str(tmp_path), "--gemini-models", "gemini-3.8-flash-confirm-me",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = bench.main()
    assert rc == 0

    report = json.loads((tmp_path / f"{bench.dt.date.today().isoformat()}.json").read_text())
    row = report["models"]["gemini-3.8-flash-confirm-me"]
    assert row["provider"] == "gemini"
    for probe_name in bench.probes():
        assert "latency_s" in row[probe_name]
        assert "tokens" in row[probe_name]

    md = (tmp_path / f"{bench.dt.date.today().isoformat()}.md").read_text()
    assert "gemini-3.8-flash-confirm-me (Gemini API)" in md
    assert "total tokens" in md
    # The fake key is never written to the report.
    assert "fake-not-real" not in md
    assert "fake-not-real" not in json.dumps(report)


def test_dash_list_without_key_exits_1_and_never_calls_the_endpoint(monkeypatch, capsys):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--grok-list"])

    rc = bench.main()

    assert rc == 1
    assert "XAI_API_KEY unset" in capsys.readouterr().out


def test_dash_list_with_key_prints_ids_from_the_endpoint(monkeypatch, capsys):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    monkeypatch.setenv("XAI_API_KEY", "fake-not-real")
    net = FakeCompatNet({bench.XAI_DEFAULT_BASE: {"chat": "", "models": ["grok-4-fast", "grok-4"]}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--grok-list"])

    rc = bench.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "grok-4-fast" in out and "grok-4" in out
    assert "fake-not-real" not in out


def test_glm_provider_never_collides_with_the_ollama_glm_tier(monkeypatch, tmp_path):
    """bench/model_bench.py's own DEFAULT_MODELS already includes the Ollama
    tag 'glm-5.2' (a different thing: the router's Ollama-bridge GLM tier).
    The new GLM(Zhipu/Z.ai) provider must not be confused with it — same
    model string routed to the Ollama path, never to generate_openai_compat,
    unless it was explicitly listed under --glm-models."""
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)

    def ollama_net(request, timeout=None):
        return _resp({"response": "unknown", "eval_count": 1})

    monkeypatch.setattr(bench.urllib.request, "urlopen", ollama_net)
    argv = ["model_bench.py", "--models", "glm-5.2", "--baselines", "",
            "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)

    rc = bench.main()
    assert rc == 0
    report = json.loads((tmp_path / f"{bench.dt.date.today().isoformat()}.json").read_text())
    assert "provider" not in report["models"]["glm-5.2"]


# --------------------------------------------------------------------------- #
# 2026-09-26 harness fixes: workspace header, transient retry, thinking
# control, and error (inconclusive) vs FAIL on the scoreboard. None of these
# touch a probe's pass criteria — only how the call is made and how a call
# that never produced an answer is recorded.
# --------------------------------------------------------------------------- #
class _HTTP:
    """Scripted urlopen: a list of outcomes consumed in order. An int is an
    HTTPError with that status (optional headers via a (code, headers) tuple);
    a dict is a JSON body. Every request is recorded."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request, timeout=None):
        payload = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append((request.full_url, dict(request.header_items()), payload))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, tuple):
            code, headers = outcome
            raise urllib.error.HTTPError(request.full_url, code, "err", headers, io.BytesIO(b"{}"))
        if isinstance(outcome, int):
            raise urllib.error.HTTPError(request.full_url, outcome, "err", {}, io.BytesIO(b"{}"))
        return _resp(outcome)


def _chat(content, tokens=3, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"completion_tokens": tokens}}


@pytest.fixture
def no_sleep(monkeypatch):
    """Patch time.sleep (not model_bench._sleep): the tests reload the module,
    which would rebind a patched module attribute."""
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    return slept


# --- Issue 1: Anthropic workspace header ----------------------------------- #
def test_anthropic_workspace_header_only_when_env_set(monkeypatch):
    bench = reload_bench()
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    assert bench.anthropic_client_kwargs() == {}
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "  wrkspc_test  ")
    assert bench.anthropic_client_kwargs() == {
        "default_headers": {"anthropic-workspace-id": "wrkspc_test"}}


def _fake_anthropic(monkeypatch, error_text=None, reply="Carol"):
    """Install a stub `anthropic` module; records client kwargs and calls."""
    import types

    seen = {"kwargs": [], "calls": 0}

    class _Messages:
        def create(self, **request):
            seen["calls"] += 1
            if error_text:
                raise RuntimeError(error_text)
            block = types.SimpleNamespace(text=reply)
            return types.SimpleNamespace(content=[block],
                                         usage=types.SimpleNamespace(output_tokens=1))

    class _Client:
        def __init__(self, **kwargs):
            seen["kwargs"].append(kwargs)
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Client))
    return seen


def test_generate_anthropic_sends_workspace_header(monkeypatch):
    bench = reload_bench()
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    seen = _fake_anthropic(monkeypatch)
    text, _, _ = bench.generate_anthropic("claude-x", "who is shortest?")
    assert text == "Carol"
    assert seen["kwargs"] == [{"default_headers": {"anthropic-workspace-id": "wrkspc_test"}}]


WORKSPACE_400 = ("Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
                 "'message': 'This API key is not scoped to a workspace, so this request must "
                 "include the anthropic-workspace-id header'}}")


def test_workspace_400_without_header_is_a_clear_error_row_not_a_fail(monkeypatch, tmp_path, capsys):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-not-real")
    seen = _fake_anthropic(monkeypatch, error_text=WORKSPACE_400)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--models", "", "--baselines", "claude-x",
                                      "--out-dir", str(tmp_path)])

    assert bench.main() == 0
    out = capsys.readouterr().out
    assert "ANTHROPIC_WORKSPACE_ID unset" in out

    today = bench.dt.date.today().isoformat()
    row = json.loads((tmp_path / f"{today}.json").read_text())["models"]["claude-x"]
    for name in bench.probes():
        assert row[name]["status"] == "error"
        assert row[name]["pass"] is None             # inconclusive, not a FAIL
        assert row[name]["error_kind"] == "auth"
        assert row[name]["error"] == bench.WORKSPACE_HINT
    # A deterministic auth error is not retried on every probe.
    assert seen["calls"] == 1
    md = (tmp_path / f"{today}.md").read_text()
    assert "key not scoped to a workspace; set ANTHROPIC_WORKSPACE_ID or use a workspace-scoped key" in md
    assert "inconclusive" in md
    assert "fake-not-real" not in md + out


def test_workspace_400_with_header_set_says_the_header_was_rejected(monkeypatch):
    bench = reload_bench()
    kind, message = bench.classify_error(RuntimeError(WORKSPACE_400), workspace_set=True)
    assert kind == "auth"
    assert "ANTHROPIC_WORKSPACE_ID" in message and "not accepted" in message


# --- Issue 2: transient retry + Gemini thinking ---------------------------- #
def test_compat_retries_503_and_429_then_succeeds_honouring_retry_after(monkeypatch, no_sleep):
    bench = reload_bench()
    net = _HTTP([503, (429, {"Retry-After": "7"}), _chat("Carol")])
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    meta = {}
    text, _, _ = bench.generate_openai_compat("https://example.test/v1", "k", "m", "p", meta=meta)
    assert text == "Carol"
    assert len(net.requests) == 3 and meta["attempts"] == 3
    assert no_sleep[1] == 7.0                       # Retry-After honoured
    assert 0 < no_sleep[0] <= bench.RETRY_AFTER_CAP_S


def test_compat_retry_is_bounded_to_three_attempts(monkeypatch, no_sleep):
    bench = reload_bench()
    net = _HTTP([429, 429, 429, _chat("never reached")])
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    with pytest.raises(urllib.error.HTTPError) as info:
        bench.generate_openai_compat("https://example.test/v1", "k", "m", "p")
    assert info.value.code == 429
    assert len(net.requests) == bench.MAX_ATTEMPTS == 3
    assert len(no_sleep) == 2


def test_retry_after_is_capped_and_accepts_http_dates(monkeypatch):
    bench = reload_bench()
    exc = urllib.error.HTTPError("u", 429, "x", {"Retry-After": "3600"}, io.BytesIO(b""))
    assert bench._retry_after_s(exc, 1) == bench.RETRY_AFTER_CAP_S
    future = bench.email.utils.format_datetime(
        bench.dt.datetime.now(bench.dt.timezone.utc) + bench.dt.timedelta(seconds=5), usegmt=True)
    exc = urllib.error.HTTPError("u", 503, "x", {"Retry-After": future}, io.BytesIO(b""))
    assert 0 <= bench._retry_after_s(exc, 1) <= 5


def test_non_transient_errors_are_not_retried(monkeypatch, no_sleep):
    bench = reload_bench()
    net = _HTTP([401])
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    with pytest.raises(urllib.error.HTTPError):
        bench.generate_openai_compat("https://example.test/v1", "k", "m", "p", thinking_params={})
    assert len(net.requests) == 1 and no_sleep == []


def test_gemini_minimises_thinking_the_documented_way_with_headroom(monkeypatch, tmp_path, no_sleep):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-not-real")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.test/gemini")
    net = FakeCompatNet({"https://example.test/gemini": {"chat": "Carol", "tokens": 1}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--models", "", "--baselines", "",
                                      "--out-dir", str(tmp_path), "--gemini-models", "gem-x"])
    assert bench.main() == 0
    _, _, payload = net.requests[0]
    assert payload["reasoning_effort"] == "low"
    assert "enable_thinking" not in payload           # Qwen's switch is not sent to Gemini
    assert payload["max_tokens"] == bench.MAX_TOKENS + bench.THINK_HEADROOM


def test_qwen_keeps_its_enable_thinking_switch_and_plain_budget(monkeypatch, tmp_path):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    monkeypatch.setenv("QWEN_API_KEY", "fake-not-real")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.test/qwen")
    net = FakeCompatNet({"https://example.test/qwen": {"chat": "Carol", "tokens": 1}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--models", "", "--baselines", "",
                                      "--out-dir", str(tmp_path), "--qwen-models", "qwen-x"])
    assert bench.main() == 0
    _, _, payload = net.requests[0]
    assert payload["enable_thinking"] is False
    assert "reasoning_effort" not in payload
    assert payload["max_tokens"] == bench.MAX_TOKENS


def test_rejected_reasoning_effort_is_retried_once_without_it(monkeypatch, no_sleep):
    bench = reload_bench()
    net = _HTTP([400, _chat("ok")])
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    text, _, _ = bench.generate_openai_compat("https://example.test/v1", "k", "m", "p",
                                              thinking_params={"reasoning_effort": "low"})
    assert text == "ok"
    assert "reasoning_effort" in net.requests[0][2]
    assert "reasoning_effort" not in net.requests[1][2]


def test_transport_errors_are_error_rows_not_fails(monkeypatch, tmp_path, no_sleep):
    """A provider that is rate limited on every attempt of one probe is
    recorded as ERR (inconclusive) on that probe; the probes that answered are
    still scored normally. The model is never a clean sweep from this run."""
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-not-real")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.test/gemini")
    answers = {"extract": "kim@venue.com.au\nops@slushfest.com", "reason": "Carol",
               "price-honesty": "UNKNOWN", "tier-math": "2", "deep-reason": "5",
               "summarise": "Clean the machine nightly. It prevents breakdowns.",
               "code": "def is_palindrome(s):\n    t=[c.lower() for c in s if c.isalnum()]\n    return t==t[::-1]"}
    outcomes = []
    for name in bench.probes():
        outcomes += [429, 503, 429] if name == "tier-math" else [_chat(answers[name])]
    net = _HTTP(outcomes)
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--models", "", "--baselines", "",
                                      "--out-dir", str(tmp_path), "--gemini-models", "gem-x"])
    assert bench.main() == 0
    today = bench.dt.date.today().isoformat()
    row = json.loads((tmp_path / f"{today}.json").read_text())["models"]["gem-x"]
    assert row["tier-math"]["status"] == "error" and row["tier-math"]["pass"] is None
    assert row["tier-math"]["error_kind"] == "rate_limited"
    assert bench.row_counts(row) == (6, 0, 1)
    assert bench.row_verdict(row) == "inconclusive"
    md = (tmp_path / f"{today}.md").read_text()
    assert "ERR (rate_limited)" in md and "6/7 · 1 err" in md and "inconclusive" in md


def test_empty_truncated_reply_is_a_budget_error_not_a_fail(monkeypatch, no_sleep):
    bench = reload_bench()
    monkeypatch.setattr(bench.urllib.request, "urlopen", _HTTP([_chat("", finish="length")]))
    with pytest.raises(bench.BenchCallError) as info:
        bench.generate_openai_compat("https://example.test/v1", "k", "m", "p")
    assert info.value.kind == "budget"


def test_verdicts_keep_fail_apart_from_inconclusive():
    bench = reload_bench()
    ok = {"pass": True, "status": "pass", "latency_s": 1.0}
    bad = {"pass": False, "status": "fail", "latency_s": 1.0}
    err = bench.error_row("unavailable", "HTTP 503")
    legacy_err = {"pass": False, "error": "HTTP Error 410: Gone"}   # pre-fix report shape
    names = list(bench.probes())
    assert bench.row_verdict({n: ok for n in names}) == "clean sweep"
    assert bench.row_verdict({**{n: ok for n in names}, names[0]: err}) == "inconclusive"
    assert bench.row_verdict({**{n: ok for n in names}, names[0]: legacy_err}) == "inconclusive"
    assert bench.row_verdict({**{n: ok for n in names}, names[0]: err, names[1]: bad}) == "fail"


def test_same_day_merge_never_lets_an_outage_erase_a_scored_probe():
    bench = reload_bench()
    names = list(bench.probes())
    scored = {"pass": True, "status": "pass", "latency_s": 1.0}
    prior = {"m": {n: scored for n in names}}
    fresh = {"m": {**{n: {"pass": False, "status": "fail", "latency_s": 2.0} for n in names},
                   names[0]: bench.error_row("rate_limited", "HTTP 429")}}
    merged = bench.merge_rows(prior, fresh)["m"]
    assert merged[names[0]] == scored                 # outage kept the earlier evidence
    assert merged[names[1]]["status"] == "fail"       # a fresh scored answer still wins


# --- Issue 3: thinking-capable Ollama models ------------------------------- #
def test_strip_reasoning_scores_only_the_final_answer():
    bench = reload_bench()
    assert bench.strip_reasoning("<think>Alice > Bob > Carol</think>\nCarol") == "Carol"
    assert bench.strip_reasoning("The user wants the name. Alice is tallest.</think>Carol") == "Carol"
    assert bench.strip_reasoning("  Carol  ") == "Carol"
    assert bench.strip_reasoning("") == ""


class _OllamaNet:
    """Fake Ollama: /api/show returns per-model `thinking` metadata;
    /api/generate returns a reply that LEAKS reasoning unless a think level
    was requested (the observed glm-5.3 behaviour under think:false)."""

    def __init__(self, show: dict, answer="Carol", leak="The user wants me to... Alice > Bob."):
        self.show, self.answer, self.leak = show, answer, leak
        self.generate_payloads = []

    def __call__(self, request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/api/show"):
            meta = self.show.get(payload["model"])
            if meta is None:
                raise urllib.error.HTTPError(request.full_url, 404, "nf", {}, io.BytesIO(b"{}"))
            return _resp(meta)
        self.generate_payloads.append(payload)
        can_disable = False in ((self.show.get(payload["model"]) or {}).get("thinking") or {}).get("values", [False])
        if payload["think"] is False and not can_disable:
            return _resp({"response": self.leak + " " + self.answer, "eval_count": 40})
        return _resp({"response": self.answer, "thinking": self.leak, "eval_count": 40})


def test_ollama_think_level_for_models_that_cannot_disable_thinking(monkeypatch):
    bench = reload_bench()
    net = _OllamaNet({
        "levels-only": {"capabilities": ["thinking"], "thinking": {"values": ["low", "high", "max"]}},
        "can-disable": {"capabilities": ["thinking"], "thinking": {"values": [False, "high"]}},
        "bool-think": {"capabilities": ["thinking"], "thinking": {"values": [False, True]}},
        "no-meta": {"capabilities": ["completion"]},
        "mid-only": {"thinking": {"values": ["medium", "high"]}},
    })
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    base = "https://ollama.example"
    assert bench.ollama_think_setting(base, "", "levels-only") == "low"
    assert bench.ollama_think_setting(base, "", "can-disable") is False
    assert bench.ollama_think_setting(base, "", "bool-think") is False
    assert bench.ollama_think_setting(base, "", "no-meta") is False
    assert bench.ollama_think_setting(base, "", "mid-only") == "medium"
    assert bench.ollama_think_setting(base, "", "unknown-model") is False   # /api/show 404


def test_ollama_generate_scores_answer_not_reasoning_for_levels_only_model(monkeypatch):
    bench = reload_bench()
    net = _OllamaNet({"levels-only": {"thinking": {"values": ["low", "high", "max"]}},
                      "can-disable": {"thinking": {"values": [False, "high"]}}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    meta = {}
    text, _, _ = bench.generate("https://ollama.example", "", "levels-only", "p", meta=meta)
    assert text == "Carol"                              # reasoning trace not scored
    assert net.generate_payloads[-1]["think"] == "low"
    assert net.generate_payloads[-1]["options"]["num_predict"] == bench.MAX_TOKENS + bench.THINK_HEADROOM
    assert meta["think"] == "low"

    meta = {}
    bench.generate("https://ollama.example", "", "can-disable", "p", meta=meta)
    assert net.generate_payloads[-1]["think"] is False   # unchanged fast path
    assert net.generate_payloads[-1]["options"]["num_predict"] == bench.MAX_TOKENS
    assert meta["think"] is False


def test_ollama_row_records_think_mode_and_report_notes_it(monkeypatch, tmp_path):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    net = _OllamaNet({"levels-only": {"thinking": {"values": ["low", "high"]}}}, answer="2")
    monkeypatch.setattr(bench.urllib.request, "urlopen", net)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--models", "levels-only", "--baselines", "",
                                      "--out-dir", str(tmp_path), "--base", "https://ollama.example"])
    assert bench.main() == 0
    today = bench.dt.date.today().isoformat()
    row = json.loads((tmp_path / f"{today}.json").read_text())["models"]["levels-only"]
    assert row["think"] == "low"
    assert row["tier-math"]["pass"] is True             # the answer "2", not the leaked trace
    assert "think=`low`" in (tmp_path / f"{today}.md").read_text()


def test_ollama_retired_model_is_one_inconclusive_reason_not_seven_fails(monkeypatch, tmp_path):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    calls = []

    def gone(request, timeout=None):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 410, "Gone", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(bench.urllib.request, "urlopen", gone)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--models", "old-model", "--baselines", "",
                                      "--out-dir", str(tmp_path), "--base", "https://ollama.example"])
    assert bench.main() == 0
    today = bench.dt.date.today().isoformat()
    row = json.loads((tmp_path / f"{today}.json").read_text())["models"]["old-model"]
    assert {row[n]["error_kind"] for n in bench.probes()} == {"gone"}
    assert all(row[n]["pass"] is None for n in bench.probes())
    assert sum(u.endswith("/api/generate") for u in calls) == 1


# --- Endpoint privacy: Qwen base only from its secret, never published ----- #
def test_qwen_without_base_url_is_skipped_never_sent_to_a_generic_endpoint(monkeypatch, tmp_path, capsys):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    monkeypatch.setenv("QWEN_API_KEY", "fake-not-real")          # key set, base NOT set
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--models", "", "--baselines", "",
                                      "--out-dir", str(tmp_path), "--qwen-models", "qwen-x"])
    assert bench.main() == 0                    # the autouse fixture proves no HTTP call
    out = capsys.readouterr().out
    assert "QWEN_BASE_URL unset" in out and "no fallback" in out
    today = bench.dt.date.today().isoformat()
    assert json.loads((tmp_path / f"{today}.json").read_text())["models"] == {}

    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--qwen-list"])
    assert bench.main() == 1
    assert "QWEN_BASE_URL unset" in capsys.readouterr().out


def test_no_generic_qwen_endpoint_in_source():
    source = (pathlib.Path(__file__).resolve().parent.parent / "bench" / "model_bench.py").read_text()
    assert "aliyuncs.com" not in source.lower()   # no hardcoded Qwen endpoint to fall back to


def test_base_url_never_reaches_a_report_or_stdout(monkeypatch, tmp_path, capsys, no_sleep):
    bench = reload_bench()
    _clear_provider_env(monkeypatch, bench)
    private = "https://private-workspace-123.example.test/compatible-mode/v1"
    monkeypatch.setenv("QWEN_API_KEY", "fake-not-real")
    monkeypatch.setenv("QWEN_BASE_URL", private)

    def leaky(request, timeout=None):
        raise ValueError(f"unknown url type: {request.full_url}")   # urllib-style message with URL

    monkeypatch.setattr(bench.urllib.request, "urlopen", leaky)
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--models", "", "--baselines", "",
                                      "--out-dir", str(tmp_path), "--qwen-models", "qwen-x"])
    assert bench.main() == 0
    today = bench.dt.date.today().isoformat()
    blob = ((tmp_path / f"{today}.json").read_text() + (tmp_path / f"{today}.md").read_text()
            + capsys.readouterr().out)
    assert "private-workspace-123" not in blob
    assert "<redacted>" in blob

    monkeypatch.setattr(sys, "argv", ["model_bench.py", "--qwen-list"])
    bench.main()
    assert "private-workspace-123" not in capsys.readouterr().out


def test_merge_does_not_carry_scores_from_an_older_harness():
    bench = reload_bench()
    names = list(bench.probes())
    leaked_trace_fail = {"pass": False, "latency_s": 2.3, "reply_head": "The user wants me to..."}
    prior = {"m": {n: leaked_trace_fail for n in names}, "untouched": {"x": 1}}
    fresh = {"m": {n: bench.error_row("budget", "no answer") for n in names}}
    merged = bench.merge_rows(prior, fresh, same_harness=False)
    assert all(merged["m"][n]["status"] == "error" for n in names)
    assert merged["untouched"] == {"x": 1}     # models not re-run keep their rows
