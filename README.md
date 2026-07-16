# claude-model-router

Quality-first Claude model routing with cost control, current model metadata, effort settings, usage logging, and safe manual overrides.

This repo provides a small Python router for API traffic. A fast Haiku classifier chooses a tier, then the router dispatches the task to the appropriate model. Every call is logged so you can inspect tier mix, token use, and fallback behavior.

## Current model registry

| Tier | Model | API ID | Default use |
|---|---|---|---|
| `haiku` | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Mechanical extraction, cleanup, formatting, high-volume subagents |
| `sonnet` | Claude Sonnet 5 | `claude-sonnet-5` | Default coding, drafting, data analysis, tool use, and agentic work |
| `glm` | GLM 5.2 (Ollama bridge) | `glm-5.2:cloud` | Heavy NON-stakes bulk reasoning/drafting between Sonnet and Opus; `stakes=True` skips it (NUMBERS RULE) |
| `opus` | Claude Opus 4.8 | `claude-opus-4-8` | Complex architecture, large refactors, enterprise-quality analysis |
| `fable` | Claude Fable 5 | `claude-fable-5` | Frontier reserve for hardest reasoning and failed Opus cases |

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

## Setup — routing server, Ollama Cloud, offline mode (v5.1)

The router runs on two engines. `python router.py --doctor` checks both and
prints a PASS/FAIL table with the exact fix for every failing row (no tokens
spent, no secrets printed; exit 0 when at least one engine is ready).

**Anthropic engine** — `pip install anthropic` + `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`).

**Ollama bridge** — resolved as a priority chain; the first *ready* base wins:

1. `CLAUDE_ROUTER_OLLAMA_URL` — one URL **or a comma-separated list**. Point it
   at a daemon on your machine, or at a routing server on your tailnet
   (e.g. `http://<tailscale-host-or-100.x-ip>:11434` with *Expose Ollama to
   the network* enabled in the Ollama app). A daemon signed in to an Ollama
   account runs `:cloud` tags through that same endpoint — no API key needed
   on the client. Defaults to `http://localhost:11434`.
2. `https://ollama.com` — appended automatically when `OLLAMA_API_KEY` is set
   (Ollama Cloud direct; the backstop when no daemon is reachable). Note:
   ollama.com answers unauthenticated probes but rejects generation without a
   key, and doctor flags exactly that.

```bash
# Typical multi-PC tailnet setup (client side):
export CLAUDE_ROUTER_OLLAMA_URL="http://<routing-server>:11434,http://<second-pc>:11434"
export OLLAMA_API_KEY=...        # optional: Ollama Cloud as the final backstop
export GLM_OLLAMA_TAG=glm-5.2:cloud   # optional tag override (CLAUDE_ROUTER_GLM_MODEL wins over it)
python router.py --doctor
```

On Windows, `setup-windows.ps1` does the same in one command (persists the env
vars for the user, then runs doctor):

```powershell
.\setup-windows.ps1 -RoutingServer "http://<tailscale-ip>:11434" -OllamaApiKey "<key>"
```

Behaviour by environment (each engine degrades honestly, never silently):

| Anthropic | Bridge | Mode |
|---|---|---|
| ready | ready | Full ladder: `haiku -> sonnet -> glm -> opus -> fable` |
| ready | down | Claude-only: `glm` dispatch falls back to `sonnet`; escalation skips nothing else |
| missing | ready | **OFFLINE**: everything routes to the bridge, Claude tiers are skipped in escalation, `stakes=True` refuses (NUMBERS RULE — stakes never runs on the bridge) |
| missing | down | `RouterSetupError` pointing at `--doctor` |

## What changed in v5.1

- GLM 5.2 (`glm` tier) between Sonnet and Opus via the Ollama bridge, with the
  NUMBERS RULE `stakes=True` guard and one-strike escalation.
- Multi-base bridge chain: comma-separated `CLAUDE_ROUTER_OLLAMA_URL` + automatic
  Ollama Cloud backstop when `OLLAMA_API_KEY` is set; first ready base wins.
- OFFLINE (Ollama-only) mode: the router now works with no `anthropic` package
  and no key — classification defaults to the bridge, Claude tiers are skipped,
  stakes tasks are refused with a clean `RouterSetupError`.
- `python router.py --doctor` setup check and `--registry`/`--tier`/`--stakes`/
  `--effort` CLI flags.
- Env is read at call time everywhere (a long-lived server picks up key
  rotation and URL changes without a restart).

## What changed in v5.0

- Added a structured `MODEL_REGISTRY` with API IDs, labels, roles, context windows, output caps, pricing fields, and availability.
- Updated routing tiers for Haiku 4.5, Sonnet 5, Opus 4.8, and Fable 5.
- Added explicit effort control through `output_config={"effort": ...}` for supported models.
- Added manual tier override via `run(..., tier="opus")`.
- Added safe fallback from Fable access errors to a lower available tier.
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

# Ollama bridge (glm tier) — see the Setup section above
export CLAUDE_ROUTER_OLLAMA_URL="http://<routing-server>:11434"  # or a comma-separated chain
export OLLAMA_API_KEY=...                  # optional Ollama Cloud backstop
export CLAUDE_ROUTER_GLM_MODEL=glm-5.2:cloud   # wins over GLM_OLLAMA_TAG, then the registry default
export CLAUDE_ROUTER_OLLAMA_TIMEOUT=300    # generation timeout in seconds
```

Tier-specific effort overrides are also supported:

```bash
export CLAUDE_ROUTER_SONNET_EFFORT=medium
export CLAUDE_ROUTER_OPUS_EFFORT=xhigh
export CLAUDE_ROUTER_FABLE_EFFORT=high
```

## Files

- `router.py` — classifier, registry, dispatch, fallback, doctor, and usage log.
- `hq_orchestrator/` — MCP server for the tri-agent handoff protocol (see below).
- `tests/` — offline test suite (no network, no keys): `python -m pytest tests/`.
- `MODEL-ROUTING-POLICY.md` — quality-first routing policy for Claude Code / cowork sessions.
- `router-usage.csv` — generated locally at runtime; do not commit sensitive logs.
- `LICENSE` — MIT.

## Safety notes

Do not commit API keys, private transcripts, local machine paths, session IDs, or customer data. Manual model overrides are respected.

## Local Google Drive / Claude HQ sync

For a local Windows sync folder such as:

```text
<drive>:\path\to\Claude HQ\.git
```

pull the branch or merge the PR from GitHub on that machine, then let Google Drive for Desktop sync the working tree. The chat connector cannot directly write to a local drive-letter path, so GitHub is the safe source of truth for the code change.

## License

MIT licensed. Built by Elie Dagher — Snowflow NSW, Slushieco, ReGen Labs Engineering, DISPATCHIQ.

## hq-orchestrator (MCP server)

`hq_orchestrator/` implements the tri-agent handoff protocol defined in
`edagher92-coder/.github` under `orchestration/handoff/` (contract v1.0):
Fable 5 dispatches task envelopes; this server validates them, assembles the
worker prompt (system card + skill packs + context files + dependency
artifacts), calls Opus 4.8 / Sonnet 5 / Haiku with a forced `submit_result`
tool so replies always parse, validates the result envelope, and persists
everything under a runs directory.

```bash
pip install "mcp[cli]" anthropic
export ANTHROPIC_API_KEY=sk-...
export HQ_CARDS_DIR=/path/to/.github/orchestration/system-cards
export HQ_SKILLS_DIRS=/path/to/.github/claude-defaults/skills:/path/to/.github/orchestration/skills/packs
python -m hq_orchestrator.server
```

Tools: `delegate_task`, `get_task_status`, `read_artifact`, and
`route_to_automation` (refuses without a named human approver). Core logic is
stdlib-only and fully tested without network: `python -m pytest tests/test_hq_orchestrator.py`.
