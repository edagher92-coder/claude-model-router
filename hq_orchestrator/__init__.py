"""hq-orchestrator — MCP server for the tri-agent pipeline.

Fable 5 orchestrates; this server dispatches task envelopes to Opus 4.8 /
Sonnet 5 / Haiku workers and returns validated result envelopes.

Canonical envelope schemas live in edagher92-coder/.github under
orchestration/handoff/ (task-envelope.schema.json, result-envelope.schema.json,
contract version 1.0). This package implements that contract; it does not
duplicate the schema files.
"""

from .core import (
    RunStore,
    delegate,
    validate_result_envelope,
    validate_task_envelope,
)

__all__ = ["RunStore", "delegate", "validate_task_envelope", "validate_result_envelope"]
