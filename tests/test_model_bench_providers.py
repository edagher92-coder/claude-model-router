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
