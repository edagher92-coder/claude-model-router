"""Two-tier delegation tests for hq_orchestrator.core.orchestrate.

No network, no API keys: a fake caller_for parses the task_id out of the worker
message and returns a scripted result envelope, recording every system prompt it
saw so we can assert on routing and the orchestrator note.

The chain under test: Fable (the test) builds a top envelope assigned to an Opus
sub-manager (role=orchestrator); Opus returns child subtasks routed to
glm/sonnet/haiku; orchestrate() executes them cheapest-first, depth-limited.
"""
import json
import re

import pytest

from hq_orchestrator import core


@pytest.fixture
def cards_dir(tmp_path):
    d = tmp_path / "cards"
    d.mkdir()
    (d / "opus-4-8-engineer.md").write_text("You are the Opus engineer.", encoding="utf-8")
    (d / "sonnet-5-developer.md").write_text("You are the Sonnet developer.", encoding="utf-8")
    return str(d)


@pytest.fixture
def store(tmp_path):
    return core.RunStore(tmp_path / "runs-base")


def make_caller_for(script, seen):
    """script: {task_id: dict of extra result fields (e.g. subtasks, summary)}.
    seen: list that receives (model_id, system_prompt, task_id) per call."""
    def caller_for(assigned_model):
        def caller(model_id, system, message, submit_tool):
            tid = re.search(r'"task_id": "(T\d{3})"', message).group(1)
            rid = re.search(r'"run_id": "([^"]+)"', message).group(1)
            seen.append((model_id, system, tid))
            result = {
                "envelope_version": "1.0",
                "run_id": rid,
                "task_id": tid,
                "status": "completed",
                "self_check": {"verified": [], "unverified": []},
            }
            result.update(script.get(tid, {}))
            return result
        return caller
    return caller_for


def top_env(**over):
    env = {
        "envelope_version": "1.0",
        "run_id": "R1",
        "task_id": "T001",
        "objective": "Ship the feature",
        "assigned_model": "claude-opus-4-8",
        "task_type": "implementation",
        "acceptance_checks": ["it works"],
        "role": "orchestrator",
    }
    env.update(over)
    return env


def child(task_id, model, objective="do a slice", stakes=None):
    c = {
        "task_id": task_id,
        "objective": objective,
        "assigned_model": model,
        "task_type": "implementation",
        "acceptance_checks": ["done"],
    }
    if stakes is not None:
        c["stakes"] = stakes
    return c


# --------------------------------------------------------------------------- #
# Single tier (no subtasks) — orchestrate == delegate
# --------------------------------------------------------------------------- #
def test_leaf_task_returns_direct_result(cards_dir, store):
    seen = []
    caller_for = make_caller_for({"T001": {"summary": "done directly"}}, seen)
    env = top_env(role="worker")  # a plain worker, no decomposition

    result = core.orchestrate(env, caller_for, store, cards_dir=cards_dir)

    assert result["status"] == "completed"
    assert "subtask_results" not in result
    assert [s[2] for s in seen] == ["T001"]
    # role != orchestrator -> no orchestrator note in the system prompt
    assert "ORCHESTRATOR MODE" not in seen[0][1]


# --------------------------------------------------------------------------- #
# Two tiers — Opus decomposes to glm/sonnet/haiku
# --------------------------------------------------------------------------- #
def test_opus_decomposes_and_children_run_cheapest_first(cards_dir, store):
    seen = []
    script = {
        "T001": {"subtasks": [
            child("T010", "glm-5.2", "summarise the corpus"),
            child("T011", "claude-sonnet-5", "write the module"),
            child("T012", "claude-haiku-4-5", "reformat the table"),
        ]},
        "T010": {"summary": "bulk summary"},
        "T011": {"summary": "module written"},
        "T012": {"summary": "table reformatted"},
    }
    caller_for = make_caller_for(script, seen)

    result = core.orchestrate(top_env(), caller_for, store, cards_dir=cards_dir)

    # Opus got the orchestrator note; three children executed on their tiers.
    assert "ORCHESTRATOR MODE" in seen[0][1]
    ran = {tid: model for model, _system, tid in seen}
    assert ran == {
        "T001": "claude-opus-4-8",
        "T010": "glm-5.2",
        "T011": "claude-sonnet-5",
        "T012": "claude-haiku-4-5-20251001",  # WORKER_MODELS maps to the dated API id
    }
    assert len(result["subtask_results"]) == 3
    assert {r["task_id"] for r in result["subtask_results"]} == {"T010", "T011", "T012"}
    # Children were persisted under the same run.
    assert store.load_result("R1", "T011")["summary"] == "module written"


# --------------------------------------------------------------------------- #
# Three tiers + depth cap
# --------------------------------------------------------------------------- #
def test_depth_cap_stops_runaway(cards_dir, store):
    seen = []
    # T001(opus) -> T010(opus orchestrator) -> T020(opus orchestrator) -> would
    # spawn T030 at depth 2, which must NOT expand.
    script = {
        "T001": {"subtasks": [dict(child("T010", "claude-opus-4-8"), role="orchestrator")]},
        "T010": {"subtasks": [dict(child("T020", "claude-opus-4-8"), role="orchestrator")]},
        "T020": {"subtasks": [child("T030", "glm-5.2")]},
        "T030": {"summary": "should never run"},
    }
    caller_for = make_caller_for(script, seen)

    result = core.orchestrate(top_env(), caller_for, store, cards_dir=cards_dir, max_depth=2)

    ran = [tid for _m, _s, tid in seen]
    assert "T030" not in ran, "depth-2 subtasks must not expand"
    assert ran == ["T001", "T010", "T020"]
    # The unexpanded plan is surfaced honestly, not dropped silently.
    deepest = result["subtask_results"][0]["subtask_results"][0]
    assert "orchestration_notes" in deepest
    assert "max_depth" in deepest["orchestration_notes"][0]


# --------------------------------------------------------------------------- #
# NUMBERS RULE — a stakes task can never be routed to the Ollama bridge
# --------------------------------------------------------------------------- #
def test_stakes_child_on_glm_is_contained_as_failed(cards_dir, store):
    # The bad child is refused, but the run survives: sibling still executes.
    seen = []
    script = {"T001": {"subtasks": [
        child("T010", "glm-5.2", "quote the customer", stakes=True),
        child("T011", "claude-sonnet-5", "safe sibling work"),
    ]}, "T011": {"summary": "sibling done"}}
    caller_for = make_caller_for(script, seen)

    result = core.orchestrate(top_env(), caller_for, store, cards_dir=cards_dir)

    by_id = {r["task_id"]: r for r in result["subtask_results"]}
    assert by_id["T010"]["status"] == "failed" and "NUMBERS RULE" in by_id["T010"]["summary"]
    assert by_id["T011"]["status"] == "completed"
    assert "T010" not in [tid for _m, _s, tid in seen]  # never dispatched


def test_stakes_keyword_backstop_catches_forgotten_flag():
    # Sub-manager "forgot" stakes:true but the objective is money-ish.
    errs = core.validate_task_envelope({
        "envelope_version": "1.0", "run_id": "R", "task_id": "T001",
        "objective": "draft the invoice email for the Merivale refund",
        "assigned_model": "glm-5.2", "task_type": "copywriting",
        "acceptance_checks": ["c"],
    })
    assert any("keyword backstop" in e for e in errs)
    # Non-money bulk work on glm is still fine.
    assert core.validate_task_envelope({
        "envelope_version": "1.0", "run_id": "R", "task_id": "T001",
        "objective": "summarise this maintenance research corpus",
        "assigned_model": "glm-5.2", "task_type": "analysis",
        "acceptance_checks": ["c"],
    }) == []


# --------------------------------------------------------------------------- #
# Containment: role gate, status gate, duplicate ids, resume
# --------------------------------------------------------------------------- #
def test_worker_returned_subtasks_are_refused(cards_dir, store):
    # A leaf (e.g. the open-weight bridge) trying to spawn work is refused.
    seen = []
    script = {"T001": {"subtasks": [child("T010", "claude-opus-4-8", "spawn expensive work")]}}
    caller_for = make_caller_for(script, seen)

    result = core.orchestrate(top_env(role="worker"), caller_for, store, cards_dir=cards_dir)

    assert "subtask_results" not in result
    assert any("refused" in n for n in result["orchestration_notes"])
    assert [tid for _m, _s, tid in seen] == ["T001"]


def test_failed_manager_plan_is_not_executed(cards_dir, store):
    seen = []
    script = {"T001": {"status": "failed", "subtasks": [child("T010", "claude-sonnet-5")]}}
    caller_for = make_caller_for(script, seen)

    result = core.orchestrate(top_env(), caller_for, store, cards_dir=cards_dir)

    assert [tid for _m, _s, tid in seen] == ["T001"]
    assert any("not executed" in n for n in result["orchestration_notes"])


def test_duplicate_task_id_is_refused_not_overwritten(cards_dir, store):
    seen = []
    script = {"T001": {"subtasks": [
        child("T001", "claude-sonnet-5", "same id as parent"),   # collision
        child("T010", "claude-sonnet-5", "fine"),
    ]}, "T010": {"summary": "ok"}}
    caller_for = make_caller_for(script, seen)

    result = core.orchestrate(top_env(), caller_for, store, cards_dir=cards_dir)

    by_id = {r["task_id"]: r for r in result["subtask_results"]}
    assert by_id["T001"]["status"] == "failed" and "duplicate" in by_id["T001"]["summary"]
    assert by_id["T010"]["status"] == "completed"
    # Parent's own result on disk was not clobbered by the colliding child.
    assert store.load_result("R1", "T001").get("subtasks")


def test_completed_child_resumes_from_disk(cards_dir, store):
    # First run completes T010; second run must NOT re-dispatch it.
    script = {"T001": {"subtasks": [child("T010", "claude-sonnet-5")]},
              "T010": {"summary": "expensive result"}}
    seen1 = []
    core.orchestrate(top_env(), make_caller_for(script, seen1), store, cards_dir=cards_dir)
    assert "T010" in [tid for _m, _s, tid in seen1]

    seen2 = []
    result = core.orchestrate(top_env(), make_caller_for(script, seen2), store, cards_dir=cards_dir)
    assert "T010" not in [tid for _m, _s, tid in seen2], "completed child must resume from disk"
    assert result["subtask_results"][0]["summary"] == "expensive result"


def test_stakes_validation_direct():
    errs = core.validate_task_envelope({
        "envelope_version": "1.0", "run_id": "R", "task_id": "T001",
        "objective": "x", "assigned_model": "glm-5.2", "task_type": "analysis",
        "acceptance_checks": ["c"], "stakes": True,
    })
    assert any("NUMBERS RULE" in e for e in errs)
    # same task on a Claude tier is fine
    errs_ok = core.validate_task_envelope({
        "envelope_version": "1.0", "run_id": "R", "task_id": "T001",
        "objective": "x", "assigned_model": "claude-sonnet-5", "task_type": "analysis",
        "acceptance_checks": ["c"], "stakes": True,
    })
    assert errs_ok == []


def test_bad_role_rejected():
    errs = core.validate_task_envelope({
        "envelope_version": "1.0", "run_id": "R", "task_id": "T001",
        "objective": "x", "assigned_model": "claude-sonnet-5", "task_type": "analysis",
        "acceptance_checks": ["c"], "role": "boss",
    })
    assert any("role" in e for e in errs)


# --------------------------------------------------------------------------- #
# Live-found: open-weight workers mangle transport-identity fields
# --------------------------------------------------------------------------- #
def test_bridge_worker_identity_fields_are_stamped(cards_dir, store):
    """GLM returned placeholder run_id/envelope_version and 'status: success'
    live (2026-07-17). delegate() must stamp identity fields and normalise the
    bridge worker's status synonyms — content fields stay the worker's own."""
    def caller_for(model):
        def mangled(model_id, system, message, tool):
            return {"envelope_version": "x.y", "run_id": "WRONG", "task_id": "T999",
                    "status": "success", "summary": "real content",
                    "self_check": {"verified": [], "unverified": []}}
        return mangled

    env = {"envelope_version": "1.0", "run_id": "R1", "task_id": "T001",
           "objective": "summarise the corpus", "assigned_model": "glm-5.2",
           "task_type": "analysis", "acceptance_checks": ["done"], "role": "worker"}
    result = core.orchestrate(env, caller_for, store, cards_dir=cards_dir)
    assert result["run_id"] == "R1" and result["task_id"] == "T001"
    assert result["status"] == "completed" and result["summary"] == "real content"


# --------------------------------------------------------------------------- #
# Second-pass SEVERE: route_to_automation webhook SSRF + HMAC
# --------------------------------------------------------------------------- #
def test_validate_webhook_url_matrix():
    allow = {"hooks.zapier.com", "api.elevenlabs.io"}
    ok = core.validate_webhook_url("https://hooks.zapier.com/x", allow_hosts=allow)
    assert ok[0]
    # http (not https) refused
    assert not core.validate_webhook_url("http://hooks.zapier.com/x", allow_hosts=allow)[0]
    # host not allowlisted refused
    assert not core.validate_webhook_url("https://attacker.com/x", allow_hosts=allow)[0]
    # empty allowlist refuses everything (fail closed)
    assert not core.validate_webhook_url("https://hooks.zapier.com/x", allow_hosts=set())[0]
    # an internal IP literal is refused even if someone allowlists it
    assert not core.validate_webhook_url("https://169.254.169.254/x", allow_hosts={"169.254.169.254"})[0]
    assert not core.validate_webhook_url("https://127.0.0.1/x", allow_hosts={"127.0.0.1"})[0]
    assert not core.validate_webhook_url("https://10.0.0.9/x", allow_hosts={"10.0.0.9"})[0]


def test_sign_payload_hmac():
    body = b'{"a":1}'
    sig = core.sign_payload(body, "topsecret")
    assert sig.startswith("sha256=")
    import hashlib as _h, hmac as _m
    assert sig == "sha256=" + _m.new(b"topsecret", body, _h.sha256).hexdigest()
    assert core.sign_payload(body, "") == ""   # no secret -> unsigned


# --------------------------------------------------------------------------- #
# Kimi 4th-pass adjudicated fixes (orchestrator internals, 2026-07-17)
# --------------------------------------------------------------------------- #
def test_webhook_rejects_hostname_resolving_internal(monkeypatch):
    import socket as _s
    allow = {"rebind.evil.example"}
    # allowlisted hostname, but DNS says it points at the metadata service
    monkeypatch.setattr(core.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("169.254.169.254", 0))])
    ok, reason = core.validate_webhook_url("https://rebind.evil.example/hook", allow_hosts=allow)
    assert not ok and "internal" in reason.lower()
    # same host resolving to a normal public IP is allowed
    monkeypatch.setattr(core.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 0))])
    assert core.validate_webhook_url("https://rebind.evil.example/hook", allow_hosts=allow)[0]


def test_get_task_status_style_validation():
    # run_id / task_id validation logic (mirrors server.get_task_status guard)
    import re as _re
    assert _re.match(r"^[A-Za-z0-9._-]+$", "R1") and ".." not in "R1"
    assert not (_re.match(r"^[A-Za-z0-9._-]+$", "../../etc") and ".." not in "../../etc")
    assert core.TASK_ID_RE.match("T001")
    assert not core.TASK_ID_RE.match("../../../etc/passwd")


def test_ollama_caller_dispatch_guard(monkeypatch):
    from hq_orchestrator import ollama_caller
    # a bare name resolving to a public IP is refused at dispatch
    monkeypatch.setattr(ollama_caller.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 0))])
    import pytest
    with pytest.raises(RuntimeError, match="SSRF guard"):
        ollama_caller._dispatch_ssrf_ok("http://sneaky:11434")
    # resolves private -> allowed
    monkeypatch.setattr(ollama_caller.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("10.0.0.9", 0))])
    ollama_caller._dispatch_ssrf_ok("http://gpu-box:11434")  # no raise
    # metadata IP refused
    monkeypatch.setattr(ollama_caller.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("169.254.169.254", 0))])
    assert ollama_caller._resolves_private("evil") is False
    ollama_caller._dispatch_ssrf_ok("https://ollama.com")  # intentional public, no raise
