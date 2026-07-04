# claude-model-router

Quality-first Claude model routing with cost control, current model metadata, effort settings, usage logging, and safe manual overrides.

This repo provides a small Python router for API traffic. A fast Haiku classifier chooses a tier, then the router dispatches the task to the appropriate model. Every call is logged so you can inspect tier mix, token use, and fallback behavior.

## Current model registry

| Tier | Model | API ID | Default use |
|---|---|---|---|
| `haiku` | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Mechanical extraction, cleanup, formatting, high-volume subagents |
| `sonnet` | Claude Sonnet 5 | `claude-sonnet-5` | Default coding, drafting, data analysis, tool use, and agentic work |
| `opus` | Claude Opus 4.8 | `claude-opus-4-8` | Complex architecture, large refactors, enterprise-quality analysis |
| `fable` | Claude Fable 5 | `claude-fable-5` | Frontier reserve for hardest reasoning and failed Opus cases |
| `mythos` | Claude Mythos 5 | `claude-mythos-5` | Manual-only approved defensive cybersecurity workflows |

The router keeps Mythos disabled unless explicitly enabled with `CLAUDE_ROUTER_ENABLE_MYTHOS=true`.

## Quick start

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python router.py "Summarize this repository and propose next steps"
```

Python usage:

```python
from router import run

reply = run(
    "Refactor this FastAPI endpoint and explain the trade-offs",
    tier="sonnet",      # optional manual override
    effort="high",     # low, medium, high, xhigh, max where supported
    max_tokens=4096,
)
print(reply)
```

## What changed in v4.0

- Added a structured `MODEL_REGISTRY` with API IDs, labels, roles, context windows, output caps, pricing fields, and availability.
- Updated routing tiers for Haiku 4.5, Sonnet 5, Opus 4.8, Fable 5, and Mythos 5.
- Added explicit effort control through `output_config={"effort": ...}` for supported models.
- Added manual tier override via `run(..., tier="opus")`.
- Added safe fallback from Fable/Mythos access errors to a lower available tier.
- Added environment-variable model overrides for account-specific aliases.
- Improved CSV logging with timestamp, tier, model ID, effort, tokens, and status.

## Environment overrides

```bash
export CLAUDE_ROUTER_EFFORT=high
export CLAUDE_ROUTER_LOG=router-usage.csv

export CLAUDE_ROUTER_HAIKU_MODEL=claude-haiku-4-5-20251001
export CLAUDE_ROUTER_SONNET_MODEL=claude-sonnet-5
export CLAUDE_ROUTER_OPUS_MODEL=claude-opus-4-8
export CLAUDE_ROUTER_FABLE_MODEL=claude-fable-5
export CLAUDE_ROUTER_MYTHOS_MODEL=claude-mythos-5

# Disabled by default. Enable only for approved access/governance.
export CLAUDE_ROUTER_ENABLE_MYTHOS=false
```

Tier-specific effort overrides are also supported:

```bash
export CLAUDE_ROUTER_SONNET_EFFORT=medium
export CLAUDE_ROUTER_OPUS_EFFORT=xhigh
export CLAUDE_ROUTER_FABLE_EFFORT=high
```

## Files

- `router.py` — classifier, registry, dispatch, fallback, and usage log.
- `MODEL-ROUTING-POLICY.md` — quality-first routing policy for Claude Code / cowork sessions.
- `router-usage.csv` — generated locally at runtime; do not commit sensitive logs.
- `LICENSE` — MIT.

## Safety notes

Do not commit API keys, private transcripts, local machine paths, session IDs, or customer data. Manual model overrides are respected. Mythos is not auto-routed and remains disabled unless explicitly enabled.

## Local Google Drive / Claude HQ sync

For a local Windows sync folder such as:

```text
H:\My Drive\Claude HQ\.git
```

pull the branch or merge the PR from GitHub on that machine, then let Google Drive for Desktop sync the working tree. The chat connector cannot directly write to a local `H:` drive path, so GitHub is the safe source of truth for the code change.

## License

MIT licensed. Built by Elie Dagher — Snowflow NSW, Slushieco, ReGen Labs Engineering, DISPATCHIQ.
