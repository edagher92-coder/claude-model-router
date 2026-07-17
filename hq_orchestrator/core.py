"""Core delegation logic — stdlib only, model caller injected.

Everything here is testable without `anthropic` or `mcp` installed; the real
API caller and MCP wiring live in server.py.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Callable, Optional

ENVELOPE_VERSION = "1.0"
TASK_ID_RE = re.compile(r"^T[0-9]{3}$")

# Two-tier delegation depth cap. Fable (the session) dispatches the top task to
# an Opus sub-manager at depth 0; the sub-manager may return child subtasks that
# run at depth 1; those may sub-delegate once more at depth 2, then it stops.
# This is a hard runaway backstop, NOT a target — most work is one or two tiers.
MAX_ORCHESTRATION_DEPTH = 2

ENVELOPE_ROLES = {"worker", "orchestrator"}

# Money-ish objective wording that must never reach the Ollama bridge even when
# the envelope forgot stakes:true (NUMBERS RULE keyword backstop).
_STAKES_HINT_RE = re.compile(
    r"(?i)\b(price|pricing|quote|quoting|invoice|refund|deposit|payment|charge|GST|legal|contract)\b|\$"
)

# Appended to a sub-manager's system prompt when role == "orchestrator": it may
# either do the task or decompose it into child envelopes routed cheapest-first.
ORCHESTRATOR_NOTE = (
    "\n\n--- ORCHESTRATOR MODE ---\n"
    "You are a sub-manager in a two-tier delegation chain (Fable leads, you "
    "manage, workers execute). If this task is large, DECOMPOSE it instead of "
    "doing it all yourself: return status 'completed' with a 'subtasks' array of "
    "child task envelopes, each with task_id (T010, T011, ...), objective, "
    "assigned_model, task_type, acceptance_checks, and stakes. Route every child "
    "to the CHEAPEST capable model: glm-5.2 for heavy NON-stakes bulk "
    "(drafting/summarising/analysis), claude-sonnet-5 for normal work, "
    "claude-haiku-4-5 for mechanical transforms, claude-opus-4-8 only for the "
    "hardest pieces. NEVER route a stakes task (money, price, quote, invoice, "
    "legal, customer-facing) to glm-5.2 — mark it stakes:true and keep it on a "
    "Claude tier. If the task is small enough to finish directly, just do it and "
    "omit subtasks."
)

WORKER_MODELS = {
    # envelope value -> API model id (kept in step with router.py's registry)
    "claude-opus-4-8": "claude-opus-4-8",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    # GLM 5.2 via the Ollama bridge — mid-tier bulk reasoning between Sonnet
    # and Opus. NUMBERS RULE: never assign customer-facing price/quote/
    # invoice/legal tasks to it (orchestrator routing responsibility).
    "glm-5.2": "glm-5.2",
}

TASK_TYPES = {
    "analysis", "design", "diagnosis", "implementation",
    "copywriting", "refactor", "verification",
}

RESULT_STATUSES = {"completed", "needs_input", "failed"}

CARD_FILES = {
    "claude-opus-4-8": "opus-4-8-engineer.md",
    "claude-sonnet-5": "sonnet-5-developer.md",
}

HAIKU_CARD = (
    "You are a mechanical-transform worker in Elie Dagher's pipeline. Perform "
    "exactly the transformation requested, no judgement calls; if the task "
    "needs judgement, return status needs_input. Return a result envelope."
)

GLM_CARD = (
    "You are the mid-tier bulk-reasoning worker (GLM 5.2 via Ollama) in Elie "
    "Dagher's pipeline, sitting between Sonnet and Opus for heavy NON-stakes "
    "work: long drafting, summarising, bulk analysis, first-pass reasoning. "
    "Follow the task envelope exactly. Never invent prices, part numbers, "
    "dates, or citations; mark anything unverified in self_check.unverified. "
    "You are never the final authority on customer-facing numbers, legal, or "
    "high-stakes judgement — flag those for a Claude-tier review in "
    "self_check.concerns. Return only the structured result envelope."
)

# What a worker must return; passed to the API as a forced-choice tool schema
# so the reply always parses.
SUBMIT_RESULT_TOOL = {
    "name": "submit_result",
    "description": "Return the completed result envelope for this task.",
    "input_schema": {
        "type": "object",
        "required": ["envelope_version", "run_id", "task_id", "status", "self_check"],
        "properties": {
            "envelope_version": {"const": ENVELOPE_VERSION},
            "run_id": {"type": "string"},
            "task_id": {"type": "string", "pattern": TASK_ID_RE.pattern},
            "status": {"enum": sorted(RESULT_STATUSES)},
            "summary": {"type": "string"},
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "content_type": {"type": "string"},
                    },
                },
            },
            "self_check": {
                "type": "object",
                "required": ["verified", "unverified"],
                "properties": {
                    "verified": {"type": "array"},
                    "unverified": {"type": "array"},
                    "assumptions": {"type": "array"},
                    "concerns": {"type": "array"},
                },
            },
            "blocking_questions": {"type": "array"},
            "usage_note": {"type": "string"},
            # Two-tier delegation: a sub-manager (role=orchestrator) may return
            # child task envelopes instead of doing the work itself. The
            # orchestrator executes each one cheapest-first, depth-limited.
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["task_id", "objective", "assigned_model", "task_type", "acceptance_checks"],
                    "properties": {
                        "task_id": {"type": "string", "pattern": TASK_ID_RE.pattern},
                        "objective": {"type": "string"},
                        "assigned_model": {"enum": sorted(WORKER_MODELS)},
                        "task_type": {"enum": sorted(TASK_TYPES)},
                        "acceptance_checks": {"type": "array", "items": {"type": "string"}},
                        "stakes": {"type": "boolean"},
                        "role": {"enum": sorted(ENVELOPE_ROLES)},
                        "skills": {"type": "array"},
                        "constraints": {"type": "array"},
                        "context_files": {"type": "array"},
                        "input_artifacts": {"type": "array"},
                    },
                },
            },
        },
    },
}

# caller(model_api_id, system_prompt, user_message, submit_tool) -> dict
ModelCaller = Callable[[str, str, str, dict], dict]


def validate_task_envelope(env: dict) -> list[str]:
    """Return a list of contract violations; empty list means valid."""
    errors: list[str] = []
    if env.get("envelope_version") != ENVELOPE_VERSION:
        errors.append(f"envelope_version must be '{ENVELOPE_VERSION}'")
    for field in ("run_id", "objective"):
        if not isinstance(env.get(field), str) or not env.get(field):
            errors.append(f"'{field}' must be a non-empty string")
    if not TASK_ID_RE.match(str(env.get("task_id", ""))):
        errors.append("'task_id' must match T000 pattern")
    if env.get("assigned_model") not in WORKER_MODELS:
        errors.append(f"'assigned_model' must be one of {sorted(WORKER_MODELS)}")
    if env.get("task_type") not in TASK_TYPES:
        errors.append(f"'task_type' must be one of {sorted(TASK_TYPES)}")
    checks = env.get("acceptance_checks")
    if not isinstance(checks, list) or not checks or not all(isinstance(c, str) for c in checks):
        errors.append("'acceptance_checks' must be a non-empty list of strings")
    for field in ("context_files", "input_artifacts", "constraints", "skills"):
        value = env.get(field)
        if value is not None and not isinstance(value, list):
            errors.append(f"'{field}' must be a list when present")
    if "role" in env and env.get("role") not in ENVELOPE_ROLES:
        errors.append(f"'role' must be one of {sorted(ENVELOPE_ROLES)} when present")
    if "stakes" in env and not isinstance(env.get("stakes"), bool):
        errors.append("'stakes' must be a boolean when present")
    # NUMBERS RULE at the orchestrator layer: a stakes task never runs on the
    # Ollama bridge, no matter which tier tried to route it there.
    if env.get("stakes") and env.get("assigned_model") == "glm-5.2":
        errors.append(
            "a stakes task (money/price/quote/invoice/legal) must stay on a Claude "
            "tier — never assign it to glm-5.2 (NUMBERS RULE)"
        )
    # Keyword backstop for the same rule: a sub-manager that forgets to set
    # stakes:true on a money-ish objective still can't land it on the bridge.
    if env.get("assigned_model") == "glm-5.2" and not env.get("stakes"):
        hit = _STAKES_HINT_RE.search(str(env.get("objective", "")))
        if hit:
            errors.append(
                f"objective mentions '{hit.group(0)}' — money/quote/invoice/legal "
                "work must stay on a Claude tier (NUMBERS RULE keyword backstop)"
            )
    return errors


def validate_result_envelope(env: dict, expected_run_id: str, expected_task_id: str) -> list[str]:
    errors: list[str] = []
    if env.get("envelope_version") != ENVELOPE_VERSION:
        errors.append(f"envelope_version must be '{ENVELOPE_VERSION}'")
    if env.get("run_id") != expected_run_id:
        errors.append("run_id does not match the dispatched task")
    if env.get("task_id") != expected_task_id:
        errors.append("task_id does not match the dispatched task")
    if env.get("status") not in RESULT_STATUSES:
        errors.append(f"'status' must be one of {sorted(RESULT_STATUSES)}")
    self_check = env.get("self_check")
    if not isinstance(self_check, dict):
        errors.append("'self_check' must be an object")
    else:
        for field in ("verified", "unverified"):
            if not isinstance(self_check.get(field), list):
                errors.append(f"self_check.{field} must be a list")
    if env.get("status") == "needs_input" and not env.get("blocking_questions"):
        errors.append("status 'needs_input' requires blocking_questions")
    for artifact in env.get("artifacts") or []:
        if not isinstance(artifact, dict) or "path" not in artifact or "content" not in artifact:
            errors.append("every artifact needs 'path' and 'content'")
            break
    return errors


class RunStore:
    """Filesystem persistence for runs: envelopes, artifacts, usage ledger."""

    def __init__(self, base_dir: str | pathlib.Path):
        self.base = pathlib.Path(base_dir).resolve()

    def _run_dir(self, run_id: str) -> pathlib.Path:
        if not re.match(r"^[A-Za-z0-9._-]+$", run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        return self.base / "runs" / run_id

    def _safe_join(self, root: pathlib.Path, relative: str) -> pathlib.Path:
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError(f"path escapes the run directory: {relative!r}")
        return target

    def save_envelope(self, run_id: str, task_id: str, kind: str, envelope: dict) -> pathlib.Path:
        path = self._run_dir(run_id) / "tasks" / task_id / f"{kind}-envelope.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_result(self, run_id: str, task_id: str) -> Optional[dict]:
        path = self._run_dir(run_id) / "tasks" / task_id / "result-envelope.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_artifact(self, run_id: str, relative: str, content: str) -> pathlib.Path:
        root = self._run_dir(run_id) / "artifacts"
        path = self._safe_join(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def read_artifact(self, run_id: str, relative: str) -> str:
        root = self._run_dir(run_id) / "artifacts"
        return self._safe_join(root, relative).read_text(encoding="utf-8")

    def log_usage(self, run_id: str, record: dict) -> None:
        path = self._run_dir(run_id)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "usage.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_system_card(assigned_model: str, cards_dir: Optional[str]) -> str:
    if assigned_model == "claude-haiku-4-5":
        return HAIKU_CARD
    if assigned_model == "glm-5.2":
        return GLM_CARD
    card_file = CARD_FILES[assigned_model]
    if cards_dir:
        path = pathlib.Path(cards_dir) / card_file
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"system card {card_file} not found under cards_dir={cards_dir!r}; "
        "set HQ_CARDS_DIR to the .github checkout's orchestration/system-cards"
    )


def resolve_skills(names: list[str], skills_dirs: list[str]) -> list[tuple[str, str]]:
    """Resolve pack names to SKILL.md text; first matching dir wins (promoted
    claude-defaults/skills before staged orchestration/skills/packs)."""
    resolved: list[tuple[str, str]] = []
    for name in names:
        for base in skills_dirs:
            path = pathlib.Path(base) / name / "SKILL.md"
            if path.exists():
                resolved.append((name, path.read_text(encoding="utf-8")))
                break
        else:
            raise FileNotFoundError(f"skill pack {name!r} not found in {skills_dirs}")
    return resolved


def build_worker_message(env: dict, workspace_root: Optional[str], store: RunStore) -> str:
    """Assemble the self-contained user message: envelope, context files,
    dependency artifacts."""
    parts = ["TASK ENVELOPE:", json.dumps(env, indent=2, ensure_ascii=False)]
    root = pathlib.Path(workspace_root).resolve() if workspace_root else None
    for ref in env.get("context_files") or []:
        rel = ref["path"] if isinstance(ref, dict) else ref
        if root is None:
            raise ValueError("context_files present but no workspace_root configured")
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"context file escapes workspace root: {rel!r}")
        parts.append(f"\nCONTEXT FILE {rel}:\n{path.read_text(encoding='utf-8')}")
    for ref in env.get("input_artifacts") or []:
        content = store.read_artifact(env["run_id"], ref["path"])
        parts.append(f"\nINPUT ARTIFACT {ref['path']} (from {ref['from_task']}):\n{content}")
    parts.append(
        "\nReturn your result by calling the submit_result tool exactly once. "
        "Fill self_check honestly: 'verified' only with real evidence, "
        "everything else under 'unverified'."
    )
    return "\n".join(parts)


def delegate(
    env: dict,
    caller: ModelCaller,
    store: RunStore,
    cards_dir: Optional[str] = None,
    skills_dirs: Optional[list[str]] = None,
    workspace_root: Optional[str] = None,
) -> dict:
    """Validate, dispatch to the worker via `caller`, validate and persist the
    result envelope. Raises ValueError on contract violations."""
    errors = validate_task_envelope(env)
    if errors:
        raise ValueError("invalid task envelope: " + "; ".join(errors))

    system_parts = [load_system_card(env["assigned_model"], cards_dir)]
    for name, text in resolve_skills(env.get("skills") or [], skills_dirs or []):
        system_parts.append(f"\n--- SKILL PACK: {name} ---\n{text}")
    if env.get("role") == "orchestrator":
        system_parts.append(ORCHESTRATOR_NOTE)
    system_prompt = "\n".join(system_parts)

    store.save_envelope(env["run_id"], env["task_id"], "task", env)
    message = build_worker_message(env, workspace_root, store)

    result = caller(WORKER_MODELS[env["assigned_model"]], system_prompt, message, SUBMIT_RESULT_TOOL)

    errors = validate_result_envelope(result, env["run_id"], env["task_id"])
    if errors:
        raise ValueError("worker returned invalid result envelope: " + "; ".join(errors))

    for artifact in result.get("artifacts") or []:
        store.save_artifact(env["run_id"], artifact["path"], artifact["content"])
    store.save_envelope(env["run_id"], env["task_id"], "result", result)
    return result


# caller_for(assigned_model) -> ModelCaller. Lets orchestrate() pick the right
# engine per task (Anthropic for Claude tiers, the Ollama bridge for glm-5.2) —
# in server.py this is core.ModelCaller selection; in tests it's a fake.
CallerFor = Callable[[str], ModelCaller]


def _failed_result(run_id: str, task_id: str, error: str) -> dict:
    """Synthetic result envelope for a child that could not be dispatched —
    the run continues, the failure is honest and visible."""
    return {
        "envelope_version": ENVELOPE_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "status": "failed",
        "summary": error,
        "self_check": {"verified": [], "unverified": [f"dispatch failed: {error}"]},
    }


def orchestrate(
    env: dict,
    caller_for: CallerFor,
    store: RunStore,
    cards_dir: Optional[str] = None,
    skills_dirs: Optional[list[str]] = None,
    workspace_root: Optional[str] = None,
    depth: int = 0,
    max_depth: int = MAX_ORCHESTRATION_DEPTH,
    _seen_task_ids: Optional[set] = None,
) -> dict:
    """Two-tier delegation. Run `env` on its assigned worker; if that worker is
    a SUB-MANAGER (role='orchestrator') that completed with `subtasks`, execute
    each child cheapest-first (recursively, depth-limited) and attach the child
    results under `subtask_results`.

    The chain: Fable (the calling session) builds the top envelope, usually
    assigned to claude-opus-4-8 with role='orchestrator'. Opus decomposes into
    children routed to glm-5.2 / sonnet / haiku (rarely another opus). Every
    leaf passes the same NUMBERS-RULE-safe validation.

    Containment rules (each one exists because a review found the hole):
    - Subtasks from a NON-orchestrator worker are never executed — a leaf model
      (especially the open-weight bridge) must not be able to spawn work.
    - Subtasks from a manager whose own status is not 'completed' are ignored.
    - Duplicate task_ids are refused (they'd overwrite each other on disk).
    - One failed child doesn't abort the run: it becomes a failed result
      envelope and the siblings + parent record still persist.
    - A child with a valid completed result already on disk is not re-run
    (crash resume; the manager must re-emit the same task_ids for full reuse).
    """
    seen = _seen_task_ids if _seen_task_ids is not None else set()
    seen.add(env["task_id"])

    if depth > 0:
        prior = store.load_result(env["run_id"], env["task_id"])
        if prior and prior.get("status") == "completed" and not validate_result_envelope(
            prior, env["run_id"], env["task_id"]
        ):
            store.log_usage(env["run_id"], {
                "task_id": env["task_id"], "depth": depth, "status": "resumed_from_disk",
            })
            return prior

    caller = caller_for(env["assigned_model"])
    result = delegate(
        env, caller, store,
        cards_dir=cards_dir, skills_dirs=skills_dirs, workspace_root=workspace_root,
    )
    store.log_usage(env["run_id"], {
        "task_id": env["task_id"],
        "model": env["assigned_model"],
        "role": env.get("role", "worker"),
        "depth": depth,
        "status": result.get("status"),
        "usage_note": result.get("usage_note", ""),
    })

    subtasks = result.get("subtasks") or []
    if not subtasks:
        return result

    notes = result.setdefault("orchestration_notes", [])
    if env.get("role") != "orchestrator":
        # A plain worker (or the open-weight bridge) has no authority to spawn
        # work — record and ignore. This is the injection/cost-amplification gate.
        notes.append(
            f"{len(subtasks)} subtask(s) returned by a non-orchestrator worker — refused."
        )
        store.save_envelope(env["run_id"], env["task_id"], "orchestrated-result", result)
        return result
    if result.get("status") != "completed":
        notes.append(
            f"sub-manager status is '{result.get('status')}' — its {len(subtasks)} "
            "subtask(s) were not executed."
        )
        store.save_envelope(env["run_id"], env["task_id"], "orchestrated-result", result)
        return result
    if depth >= max_depth:
        notes.append(
            f"{len(subtasks)} subtask(s) returned at depth {depth} but max_depth "
            f"{max_depth} was reached — not expanded; flatten the plan or raise max_depth."
        )
        store.log_usage(env["run_id"], {
            "task_id": env["task_id"], "depth": depth,
            "status": "depth_capped", "subtasks_unexpanded": len(subtasks),
        })
        store.save_envelope(env["run_id"], env["task_id"], "orchestrated-result", result)
        return result

    child_results = []
    for child in subtasks:
        child = dict(child)
        child["run_id"] = env["run_id"]
        child["envelope_version"] = ENVELOPE_VERSION
        cid = str(child.get("task_id", ""))
        if cid in seen:
            child_results.append(_failed_result(
                env["run_id"], cid or "T???",
                f"duplicate task_id {cid!r} in this run — refused (would overwrite "
                "an earlier envelope on disk)",
            ))
            continue
        try:
            child_results.append(
                orchestrate(
                    child, caller_for, store,
                    cards_dir=cards_dir, skills_dirs=skills_dirs,
                    workspace_root=workspace_root,
                    depth=depth + 1, max_depth=max_depth, _seen_task_ids=seen,
                )
            )
        except (ValueError, RuntimeError) as exc:
            child_results.append(_failed_result(env["run_id"], cid or "T???", str(exc)))
    result["subtask_results"] = child_results
    store.save_envelope(env["run_id"], env["task_id"], "orchestrated-result", result)
    return result
