"""hq-orchestrator MCP server — FastMCP wiring over core.delegate.

Run:  python -m hq_orchestrator.server
Env:  ANTHROPIC_API_KEY   (required for real delegation)
      HQ_RUNS_DIR         (default .orchestrator/runs-base)
      HQ_CARDS_DIR        (path to .github/orchestration/system-cards)
      HQ_SKILLS_DIRS      (os.pathsep-separated; promoted dir first)
      HQ_WORKSPACE_ROOT   (root that context_files resolve under)

Requires: pip install "mcp[cli]" anthropic
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from . import core, ollama_caller

try:
    import anthropic
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        f"missing dependency: {exc.name}. Install with: pip install 'mcp[cli]' anthropic"
    ) from exc

mcp = FastMCP("hq-orchestrator")
_client = anthropic.Anthropic()
_store = core.RunStore(os.getenv("HQ_RUNS_DIR", ".orchestrator/runs-base"))

RETRY_DELAYS = (2, 4, 8, 16)
MAX_OUTPUT_TOKENS = 32_000


def _anthropic_caller(model: str, system: str, message: str, submit_tool: dict) -> dict:
    last_error: Exception | None = None
    for attempt, delay in enumerate((0,) + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            response = _client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=[{"role": "user", "content": message}],
                tools=[{
                    "name": submit_tool["name"],
                    "description": submit_tool["description"],
                    "input_schema": submit_tool["input_schema"],
                }],
                tool_choice={"type": "tool", "name": submit_tool["name"]},
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == submit_tool["name"]:
                    result = dict(block.input)
                    result["usage_note"] = (
                        f"input_tokens={response.usage.input_tokens} "
                        f"output_tokens={response.usage.output_tokens}"
                    )
                    return result
            raise RuntimeError("worker response contained no submit_result tool call")
        except anthropic.APIConnectionError as exc:
            last_error = exc
        except anthropic.APIStatusError as exc:
            if exc.status_code != 429 and exc.status_code < 500:
                raise  # 4xx payload/auth bugs cannot succeed on retry
            last_error = exc
    raise RuntimeError(f"worker API failed after {len(RETRY_DELAYS) + 1} attempts: {last_error}")


def _skills_dirs() -> list[str]:
    raw = os.getenv("HQ_SKILLS_DIRS", "")
    return [d for d in raw.split(os.pathsep) if d]


def _caller_for(assigned_model: str) -> core.ModelCaller:
    """Ollama-bridged tiers (GLM 5.2) go to the Ollama caller; Claude tiers
    go to the Anthropic caller."""
    if ollama_caller.is_ollama_model(assigned_model):
        return ollama_caller.call
    return _anthropic_caller


@mcp.tool()
def delegate_task(run_id: str, task_envelope: dict, timeout_seconds: int = 600) -> dict:
    """Dispatch a task envelope to a worker model and return its validated
    result envelope. Rejects contract-invalid envelopes before spending tokens."""
    del timeout_seconds  # enforced by the MCP host; kept for schema parity
    if task_envelope.get("run_id") not in (None, run_id):
        raise ValueError("run_id argument and task_envelope.run_id disagree")
    task_envelope["run_id"] = run_id
    started = time.time()
    result = core.delegate(
        task_envelope,
        caller=_caller_for(task_envelope.get("assigned_model", "")),
        store=_store,
        cards_dir=os.getenv("HQ_CARDS_DIR"),
        skills_dirs=_skills_dirs(),
        workspace_root=os.getenv("HQ_WORKSPACE_ROOT"),
    )
    _store.log_usage(run_id, {
        "task_id": task_envelope["task_id"],
        "model": task_envelope["assigned_model"],
        "status": result["status"],
        "seconds": round(time.time() - started, 1),
        "usage_note": result.get("usage_note", ""),
    })
    return result


@mcp.tool()
def orchestrate_task(run_id: str, task_envelope: dict, max_depth: int = core.MAX_ORCHESTRATION_DEPTH) -> dict:
    """Two-tier delegation: dispatch a task envelope whose worker may be a
    sub-manager (role='orchestrator', usually claude-opus-4-8). If it returns
    child subtasks, they are executed cheapest-first — Claude tiers via the
    Anthropic caller, glm-5.2 via the Ollama bridge — recursively, depth-capped.
    Stakes tasks are rejected on glm-5.2 at validation (NUMBERS RULE)."""
    if task_envelope.get("run_id") not in (None, run_id):
        raise ValueError("run_id argument and task_envelope.run_id disagree")
    task_envelope["run_id"] = run_id
    # Callers can lower the depth cap but never defeat the runaway backstop.
    max_depth = min(max_depth, 4)
    # Per-task usage rows are logged inside core.orchestrate (exactly once per
    # task, leaves included) — no duplicate summary row here.
    return core.orchestrate(
        task_envelope,
        caller_for=_caller_for,
        store=_store,
        cards_dir=os.getenv("HQ_CARDS_DIR"),
        skills_dirs=_skills_dirs(),
        workspace_root=os.getenv("HQ_WORKSPACE_ROOT"),
        max_depth=max_depth,
    )


@mcp.tool()
def get_task_status(run_id: str, task_id: str) -> dict:
    """Recover a task's state after an orchestrator restart. Prefers the
    orchestrated result (which carries subtask_results/orchestration_notes)
    over the plain worker result when both exist."""
    # Validate BEFORE building any path — run_id/task_id are MCP params and were
    # used to construct a path (orch.exists()/read_text) before _store's own
    # validation ran, a path-traversal read/probe (Kimi 4th-pass #12/#13).
    if not re.match(r"^[A-Za-z0-9._-]+$", run_id) or ".." in run_id:
        raise ValueError(f"unsafe run_id: {run_id!r}")
    if not core.TASK_ID_RE.match(task_id):
        raise ValueError(f"unsafe task_id: {task_id!r} (must match T000)")
    run_dir = _store.base / "runs" / run_id / "tasks" / task_id
    orch = run_dir / "orchestrated-result-envelope.json"
    if orch.exists():
        return {"status": "found", "result_envelope": json.loads(orch.read_text(encoding="utf-8"))}
    result = _store.load_result(run_id, task_id)
    if result is None:
        return {"status": "not_found", "result_envelope": None}
    # 'found' (not 'completed') — the wrapper must not claim completion for a
    # result whose own status is 'failed'/'needs_input' (Kimi 4th-pass #18).
    return {"status": "found", "result_envelope": result}


@mcp.tool()
def read_artifact(run_id: str, path: str) -> str:
    """Read an artifact produced by an earlier task in this run."""
    return _store.read_artifact(run_id, path)


@mcp.tool()
def route_to_automation(
    run_id: str,
    artifact_path: str,
    destination: str,
    approved_by: str,
    destination_config: dict | None = None,
) -> dict:
    """Hand a finished artifact to an external automation. REQUIRES a named
    human approver; refused otherwise. A 2xx response means ACCEPTED by the
    automation, not completed."""
    if not approved_by.strip():
        raise ValueError("route_to_automation requires a named human approver")
    if destination not in ("elevenlabs_tts", "video_generator", "zapier_webhook"):
        raise ValueError(f"unknown destination: {destination!r}")
    payload = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "artifact": _store.read_artifact(run_id, artifact_path),
        "destination": destination,
        "approved_by": approved_by,
    }
    _store.save_artifact(run_id, f"outbox/sent/{destination}-{artifact_path.replace('/', '_')}.json",
                         json.dumps(payload, indent=2, ensure_ascii=False))
    url = (destination_config or {}).get("webhook_url")
    if not url:
        return {"status": "recorded_no_webhook",
                "note": "payload written to outbox; no webhook_url in destination_config"}
    # SSRF guard: this POSTs artifact content OUT, so the target must be an https
    # host explicitly allowlisted via CLAUDE_ROUTER_WEBHOOK_ALLOW_HOSTS and must
    # not be an internal/metadata IP. Refused otherwise — the payload is already
    # safely recorded in the outbox above.
    ok, reason = core.validate_webhook_url(url)
    if not ok:
        return {"status": "webhook_refused", "reason": reason,
                "note": "payload recorded in outbox; not sent. Allowlist the host to enable delivery."}
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Idempotency-Key": f"{run_id}:{artifact_path}"}
    # Optional HMAC so the receiver can verify the POST is genuinely from us.
    signature = core.sign_payload(body, os.getenv("CLAUDE_ROUTER_WEBHOOK_SECRET", ""))
    if signature:
        headers["X-Signature"] = signature
    request = urllib.request.Request(url, data=body, headers=headers)
    last_error: Exception | None = None
    for attempt, delay in enumerate((0,) + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return {"status": "accepted_by_automation", "http_status": response.status}
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                raise RuntimeError(f"automation rejected payload (HTTP {exc.code}); not retrying") from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
    raise RuntimeError(f"automation dispatch failed after retries: {last_error}")


if __name__ == "__main__":
    mcp.run()
