"""The hq-orchestrator MCP server must stay registered and launchable.

It was written to the house standard but sat unregistered for weeks — nothing
could call it. These guards stop that regressing, and stop the registration
drifting away from the module it points at. No SDK or network needed.
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MCP_JSON = ROOT / ".mcp.json"


def _servers() -> dict:
    return json.loads(MCP_JSON.read_text(encoding="utf-8"))["mcpServers"]


def test_orchestrator_is_registered():
    assert "hq-orchestrator" in _servers(), (
        ".mcp.json must register hq-orchestrator or no session can reach it")


def test_registration_points_at_a_real_module():
    entry = _servers()["hq-orchestrator"]
    assert entry["args"][:2] == ["-m", "hq_orchestrator.server"]
    assert (ROOT / "hq_orchestrator" / "server.py").is_file()


def test_registered_module_at_least_imports_or_names_its_missing_dep():
    """Launching without the SDK must fail with a clear install line, never a
    bare traceback — the server's own import guard promises that."""
    proc = subprocess.run(
        [sys.executable, "-c", "import hq_orchestrator.server"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        combined = proc.stdout + proc.stderr
        assert "pip install" in combined, f"unhelpful failure: {combined[-300:]}"
