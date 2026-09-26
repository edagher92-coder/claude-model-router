"""Suite-wide isolation from the developer's own logins.

The router is subscription-first: on a machine where the Claude Code CLI is
signed in, it would shell out to `claude -p` and spend the real subscription.
Every test therefore starts with ROUTER_AUTH=key (the documented override) and
a CLI stand-in that refuses to run. The auth-order tests opt back in explicitly.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import subscription_auth  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_subscription(monkeypatch):
    monkeypatch.setenv("ROUTER_AUTH", "key")

    def _refuse(*_a, **_k):
        raise AssertionError("a test tried to run the real `claude -p`")

    monkeypatch.setattr(subscription_auth, "claude_print", _refuse)
    monkeypatch.setattr(subscription_auth, "_LOGIN_CACHE", {})
    yield
