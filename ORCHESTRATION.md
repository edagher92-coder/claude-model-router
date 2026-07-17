# Orchestration policy — Fable leads, Opus manages, the fleet executes

Updated: 2026-07-17. Companion to `MODEL-ROUTING-POLICY.md` (which governs
single-task tier choice); this document governs **multi-model runs**.

## The chain

```
Elie
 └─ FABLE 5 — lead orchestrator (the session)
     · owns the plan, the acceptance checks, and the final synthesis
     · never does bulk work itself; it decomposes and dispatches
     └─ OPUS 4.8 — sub-manager for each deep-tech workstream
         · takes one hard slice (architecture, gnarly debugging, security)
         · does the genuinely hard parts itself
         · decomposes the rest into child tasks, routed CHEAPEST-FIRST:
         ├─ GLM 5.2 (Ollama bridge) — heavy NON-stakes bulk: long drafts,
         │   summaries, research digests, first-pass analysis
         ├─ SONNET 5 — normal production work: most coding, edits, reviews
         └─ HAIKU 4.5 — mechanical transforms: extract, reformat, rename
```

Two ways to run it:

1. **In-session (default, works everywhere):** a Fable Claude Code session
   spawns Opus sub-agents via the Agent tool; each Opus sub-agent spawns
   Sonnet/Haiku sub-agents, and reaches GLM through router v5.1
   (`run(task, tier="glm")`). Use the `/orchestrate` skill.
2. **Server (durable, resumable):** `hq_orchestrator` runs on a PC with the
   API key. Fable dispatches a task envelope with `role: "orchestrator"`
   assigned to `claude-opus-4-8`; `core.orchestrate()` executes the returned
   child envelopes recursively, persisting every envelope/artifact to disk.
   Depth-capped at `MAX_ORCHESTRATION_DEPTH` (2) as a runaway backstop.

## Hard rules (non-negotiable)

- **NUMBERS RULE / stakes gate:** money, prices, quotes, invoices, legal, and
  customer-facing output NEVER run on the Ollama bridge. Envelope validation
  rejects `stakes: true` + `glm-5.2` outright; in-session, pass
  `stakes=True` to the router. Stakes work belongs to Opus (or Fable).
- **Skills pass down.** Workers have no native skill discovery: every
  delegation attaches the relevant SKILL.md content (envelope `skills` packs
  or pasted into the sub-agent prompt). Smallest relevant set, never the
  whole registry.
- **Honest envelopes.** Workers return `self_check` with `verified` only for
  evidence-backed claims; Fable's synthesis treats everything else as
  unverified. `needs_input` requires `blocking_questions`.
- **On-demand, not always-on.** Orchestration runs when Elie starts it. No
  standing loops, no cron-driven Claude runs without an explicit budget
  decision (token-efficiency policy).
- **One-strike escalation still applies inside a tier** (router v5.1): a
  failed/incomplete worker attempt climbs the ladder rather than retrying
  the same losing configuration.

## Choosing the worker (strong suits)

Weekly evidence beats vibes: `bench/model_bench.py` probes the live fleet on
extract / summarise / code / reasoning / **price-honesty** and writes a dated
report to `bench/reports/`. Route by the latest report, not by memory —
particularly the price-honesty column, which decides whether a model can be
trusted anywhere near customer-facing drafting (even non-stakes).

**Auto-allocation (added 2026-07-17):** the router picks the glm tier's model
itself from the latest committed report — clean sweep on all probes (including
the two business-critical ones: price-honesty and tier-math), lowest average
latency wins. Weekly loop: Monday's bench commits a fresh report → every
machine that pulls re-allocates automatically. Env overrides
(`CLAUDE_ROUTER_GLM_MODEL` > `GLM_OLLAMA_TAG`) always win; disable with
`CLAUDE_ROUTER_AUTO_ALLOCATE=0`; `--doctor` shows the active allocation and why.

Baseline mapping (see the latest report for current truth):

| Work | First choice | Notes |
|---|---|---|
| Long summaries, digests, first drafts | bench winner (2026-07-17: **GLM 5.2**) | free quota-wise; never final authority |
| Bulk non-stakes code drafts | kimi-k2.7-code | clean sweep incl. tier-math; code specialist |
| Most coding, reviews, agentic steps | Sonnet 5 | the workhorse |
| Mechanical extract/reformat | Haiku 4.5 | fast + cheap |
| Architecture, security, hard debugging | Opus 4.8 | also the sub-manager tier |
| Plan, synthesis, final judgement | Fable 5 | the session itself |

Measured caveats from the 2026-07-17 run (re-check weekly):
- **nemotron-3-nano:30b answered "1" on tier-math — it rounds DOWN** (would
  underquote). Mechanical transforms only; never near quoting logic.
- **deepseek-v4-pro / mistral-large-3:675b** got the maths right but ignored
  "answer with just the number" until token-capped — weak instruction-following
  disqualifies them from structured-envelope pipeline work for now.
- **qwen3.5:397b** — fastest overall but returned one empty reply (21s stall);
  watch flakiness before promoting it.

## Cadence

- **Weekly:** run the model bench (manually or via `model-bench.yml` once the
  `OLLAMA_API_KEY` secret is set), skim the report, adjust the mapping above
  if a model's strong suits shifted, and check `ollama.com` for new/retired
  cloud models (`/api/tags` — the bench lists what it saw).
- After any Anthropic model lineup change, update `MODEL_REGISTRY` in
  `router.py` and this table together.
