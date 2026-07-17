import importlib
import os


def load_router(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_ROUTER_AUTO_ALLOCATE", "0")
    import router

    return importlib.reload(router)


def test_registry_contains_current_tiers(monkeypatch):
    router = load_router(monkeypatch)
    data = router.registry()

    assert set(data) == {"haiku", "sonnet", "glm", "opus", "fable"}
    assert data["haiku"]["active_api_id"] == "claude-haiku-4-5-20251001"
    assert data["sonnet"]["active_api_id"] == "claude-sonnet-5"
    assert data["glm"]["active_api_id"] == "glm-5.2:cloud"
    assert data["opus"]["active_api_id"] == "claude-opus-4-8"
    assert data["fable"]["active_api_id"] == "claude-fable-5"


def test_glm_tier_shape(monkeypatch):
    router = load_router(monkeypatch)
    glm = router.registry()["glm"]

    assert glm["engine"] == "ollama"
    assert glm["availability"] == "ollama-bridge"
    assert glm["supports_effort"] is False
    assert router.registry()["sonnet"]["engine"] == "anthropic"


def test_glm_sits_between_sonnet_and_opus(monkeypatch):
    router = load_router(monkeypatch)

    assert router.ESCALATE["sonnet"] == "glm"
    assert router.ESCALATE["glm"] == "opus"
    assert router.FALLBACK["glm"] == "sonnet"
    assert router._normalise_tier("glm") == "glm"


def test_stakes_guard_skips_glm(monkeypatch):
    router = load_router(monkeypatch)

    assert router._apply_stakes("glm", stakes=True) == "sonnet"
    assert router._apply_stakes("glm", stakes=False) == "glm"
    assert router._apply_stakes("opus", stakes=True) == "opus"


def test_glm_effort_is_none(monkeypatch):
    router = load_router(monkeypatch)

    assert router._effort_for("glm", "max") is None


def test_environment_model_override(monkeypatch):
    router = load_router(monkeypatch)
    monkeypatch.setenv("CLAUDE_ROUTER_SONNET_MODEL", "custom-sonnet-id")

    assert router.registry()["sonnet"]["active_api_id"] == "custom-sonnet-id"


def test_effort_rules(monkeypatch):
    router = load_router(monkeypatch)

    assert router._effort_for("haiku", "max") is None
    assert router._effort_for("sonnet", None) == "high"
    assert router._effort_for("opus", "xhigh") == "xhigh"


def test_invalid_tier_and_effort_are_rejected(monkeypatch):
    router = load_router(monkeypatch)

    try:
        router._normalise_tier("bad")
    except ValueError as exc:
        assert "unknown tier" in str(exc)
    else:
        raise AssertionError("Expected invalid tier to raise ValueError")

    try:
        router._effort_for("sonnet", "ultracode")
    except ValueError as exc:
        assert "unsupported effort" in str(exc)
    else:
        raise AssertionError("Expected invalid effort to raise ValueError")


def test_incomplete_detection(monkeypatch):
    router = load_router(monkeypatch)

    assert router._is_incomplete("", None)
    assert router._is_incomplete("short", None)          # short, no terminator
    assert not router._is_incomplete("Done.", None)      # terse but complete (Kimi #9)
    assert router._is_incomplete("a perfectly long answer that got cut off mid", "max_tokens")
    assert router._is_incomplete("I can't help with that request as stated, because", "end_turn")
    assert not router._is_incomplete("Here is a complete, confident answer to the task.", "end_turn")


def test_one_strike_escalation_walks_the_ladder(monkeypatch):
    router = load_router(monkeypatch)

    assert router._next_tier_up("sonnet", {"sonnet"}, stakes=False) == "glm"
    assert router._next_tier_up("glm", {"sonnet", "glm"}, stakes=False) == "opus"
    assert router._next_tier_up("fable", {"fable"}, stakes=False) is None


def test_escalation_skips_glm_when_stakes(monkeypatch):
    router = load_router(monkeypatch)

    assert router._next_tier_up("sonnet", {"sonnet"}, stakes=True) == "opus"


def test_no_sonnet_glm_ping_pong(monkeypatch):
    router = load_router(monkeypatch)

    # sonnet weak -> glm; glm bridge fails; sonnet already tried -> must go UP
    tried = {"sonnet", "glm"}
    recovery = "sonnet" if "sonnet" not in tried else router._next_tier_up("glm", tried, False)
    assert recovery == "opus"
