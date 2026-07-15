"""Contract tests for hq_orchestrator.core — no network, no anthropic/mcp."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hq_orchestrator import core


def make_envelope(**overrides):
    envelope = {
        "envelope_version": "1.0",
        "run_id": "run-test",
        "task_id": "T001",
        "assigned_model": "claude-sonnet-5",
        "task_type": "copywriting",
        "objective": "Write the thing.",
        "acceptance_checks": ["thing is written"],
    }
    envelope.update(overrides)
    return envelope


def make_result(**overrides):
    result = {
        "envelope_version": "1.0",
        "run_id": "run-test",
        "task_id": "T001",
        "status": "completed",
        "summary": "done",
        "artifacts": [{"path": "out.md", "content": "hello"}],
        "self_check": {"verified": [], "unverified": []},
    }
    result.update(overrides)
    return result


class TestTaskEnvelopeValidation:
    def test_valid_envelope_passes(self):
        assert core.validate_task_envelope(make_envelope()) == []

    def test_bad_task_id_rejected(self):
        errors = core.validate_task_envelope(make_envelope(task_id="task-1"))
        assert any("task_id" in e for e in errors)

    def test_unknown_model_rejected(self):
        errors = core.validate_task_envelope(make_envelope(assigned_model="gpt-9"))
        assert any("assigned_model" in e for e in errors)

    def test_empty_acceptance_checks_rejected(self):
        errors = core.validate_task_envelope(make_envelope(acceptance_checks=[]))
        assert any("acceptance_checks" in e for e in errors)


class TestResultEnvelopeValidation:
    def test_valid_result_passes(self):
        assert core.validate_result_envelope(make_result(), "run-test", "T001") == []

    def test_mismatched_task_id_rejected(self):
        errors = core.validate_result_envelope(make_result(task_id="T002"), "run-test", "T001")
        assert any("task_id" in e for e in errors)

    def test_needs_input_requires_blocking_questions(self):
        errors = core.validate_result_envelope(
            make_result(status="needs_input"), "run-test", "T001"
        )
        assert any("blocking_questions" in e for e in errors)


class TestRunStore:
    def test_artifact_roundtrip(self, tmp_path):
        store = core.RunStore(tmp_path)
        store.save_artifact("run-test", "a/b.md", "content")
        assert store.read_artifact("run-test", "a/b.md") == "content"

    def test_path_traversal_blocked(self, tmp_path):
        store = core.RunStore(tmp_path)
        with pytest.raises(ValueError):
            store.save_artifact("run-test", "../../escape.md", "nope")

    def test_unsafe_run_id_blocked(self, tmp_path):
        store = core.RunStore(tmp_path)
        with pytest.raises(ValueError):
            store.save_artifact("../evil", "a.md", "nope")


class TestDelegate:
    def _cards_dir(self, tmp_path):
        cards = tmp_path / "cards"
        cards.mkdir()
        (cards / "sonnet-5-developer.md").write_text("You are the developer.")
        (cards / "opus-4-8-engineer.md").write_text("You are the engineer.")
        return str(cards)

    def test_happy_path_persists_everything(self, tmp_path):
        store = core.RunStore(tmp_path / "store")
        captured = {}

        def fake_caller(model, system, message, tool):
            captured["model"] = model
            captured["system"] = system
            captured["message"] = message
            return make_result()

        result = core.delegate(
            make_envelope(), fake_caller, store, cards_dir=self._cards_dir(tmp_path)
        )
        assert result["status"] == "completed"
        assert captured["model"] == "claude-sonnet-5"
        assert "You are the developer." in captured["system"]
        assert "TASK ENVELOPE:" in captured["message"]
        assert store.read_artifact("run-test", "out.md") == "hello"
        assert store.load_result("run-test", "T001")["status"] == "completed"

    def test_invalid_envelope_never_calls_model(self, tmp_path):
        store = core.RunStore(tmp_path / "store")
        calls = []
        with pytest.raises(ValueError, match="invalid task envelope"):
            core.delegate(
                make_envelope(task_type="vibes"),
                lambda *a: calls.append(a) or make_result(),
                store,
                cards_dir=self._cards_dir(tmp_path),
            )
        assert calls == []

    def test_invalid_worker_result_raises(self, tmp_path):
        store = core.RunStore(tmp_path / "store")
        with pytest.raises(ValueError, match="invalid result envelope"):
            core.delegate(
                make_envelope(),
                lambda *a: make_result(status="finished-ish"),
                store,
                cards_dir=self._cards_dir(tmp_path),
            )

    def test_skills_inlined_promoted_dir_wins(self, tmp_path):
        store = core.RunStore(tmp_path / "store")
        promoted = tmp_path / "promoted" / "mcp-forge"
        staged = tmp_path / "staged" / "mcp-forge"
        promoted.mkdir(parents=True)
        staged.mkdir(parents=True)
        (promoted / "SKILL.md").write_text("PROMOTED PACK")
        (staged / "SKILL.md").write_text("STAGED PACK")
        captured = {}

        def fake_caller(model, system, message, tool):
            captured["system"] = system
            return make_result()

        core.delegate(
            make_envelope(skills=["mcp-forge"]),
            fake_caller,
            store,
            cards_dir=self._cards_dir(tmp_path),
            skills_dirs=[str(promoted.parent), str(staged.parent)],
        )
        assert "PROMOTED PACK" in captured["system"]
        assert "STAGED PACK" not in captured["system"]

    def test_dependency_artifact_inlined(self, tmp_path):
        store = core.RunStore(tmp_path / "store")
        store.save_artifact("run-test", "T001-fix.md", "the opus fix")
        captured = {}

        def fake_caller(model, system, message, tool):
            captured["message"] = message
            return make_result(task_id="T002")

        core.delegate(
            make_envelope(
                task_id="T002",
                input_artifacts=[{"from_task": "T001", "path": "T001-fix.md"}],
            ),
            fake_caller,
            store,
            cards_dir=self._cards_dir(tmp_path),
        )
        assert "the opus fix" in captured["message"]
