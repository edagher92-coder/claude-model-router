# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## Model routing policy

This repo follows the **Claude HQ model-routing standard**. See
[`MODEL-ROUTING-POLICY.md`](./MODEL-ROUTING-POLICY.md) at the repo root
(v4.0 — Quality first, zero waste):

- **Sonnet 5** (`claude-sonnet-5`) — default workhorse for coding, drafting, tool use.
- **Haiku 4.5** — mechanical extraction, formatting, high-volume subagents.
- **Opus 4.8** — complex architecture, large refactors, high-stakes / customer-facing work.
- **Fable 5** — frontier reserve; only when a task clearly exceeds Opus.
- **Mythos 5** — manual-only, approved defensive-security workflows; never auto-route.

Any repo code that calls the Anthropic API directly should route through
`router.run(...)` from `router.py` (vendored from `edagher92-coder/claude-model-router`)
where practical.
