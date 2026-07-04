import importlib
import os


def load_router(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    import router

    return importlib.reload(router)


def test_registry_contains_current_tiers(monkeypatch):
    router = load_router(monkeypatch)
    data = router.registry()

    assert set(data) == {"haiku", "sonnet", "opus", "fable", "mythos"}
    assert data["haiku"]["active_api_id"] == "claude-haiku-4-5-20251001"
    assert data["sonnet"]["active_api_id"] == "claude-sonnet-5"
    assert data["opus"]["active_api_id"] == "claude-opus-4-8"
    assert data["fable"]["active_api_id"] == "claude-fable-5"
    assert data["mythos"]["active_api_id"] == "claude-mythos-5"
    assert data["mythos"]["enabled"] is False


def test_environment_model_override(monkeypatch):
    router = load_router(monkeypatch)
    monkeypatch.setenv("CLAUDE_ROUTER_SONNET_MODEL", "custom-sonnet-id")

    assert router.registry()["sonnet"]["active_api_id"] == "custom-sonnet-id"


def test_mythos_requires_explicit_enable(monkeypatch):
    router = load_router(monkeypatch)

    assert router.registry()["mythos"]["enabled"] is False
    monkeypatch.setenv("CLAUDE_ROUTER_ENABLE_MYTHOS", "true")
    assert router.registry()["mythos"]["enabled"] is True


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
