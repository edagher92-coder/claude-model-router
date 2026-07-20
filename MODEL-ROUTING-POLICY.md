# Model Routing Policy v5.0 — Quality First, Zero Waste

Updated: 2026-07-08

Principle: route every task to the lowest-cost Claude tier that should meet the required quality bar. Never let cost degrade quality. Never burn a premium tier on work a lower tier handles reliably. Manual model selection always wins.

## Current tiers

### 1. Haiku 4.5 — mechanical only

API ID: `claude-haiku-4-5-20251001`

Use for:
- Reformatting, renames, extraction, table cleanup, boilerplate, short transforms.
- High-volume low-latency sub-agent work.
- One-line answers and deterministic text operations.

Do not use for real reasoning, multi-step planning, customer-facing advice, architecture, financial/legal/compliance output, or large refactors.

### 2. Sonnet 5 — default workhorse

API ID: `claude-sonnet-5`

Use for:
- General coding, drafting, data analysis, tool use, agentic workflows, browser/terminal automation, and long-context work.
- Most Claude Code / cowork sessions.
- Production tasks where speed, price, and intelligence all matter.

Effort guidance:
- `high` is the default and best starting point for complex work.
- `medium` is the cost-saving default for routine agentic workflows after evals prove quality holds.
- `low` is for high-volume or latency-sensitive non-critical work.
- `xhigh` is for hard coding/agentic tasks; if the task stays quality-critical, escalate to Opus instead of repeatedly over-prompting.
- `max` is reserved for rare tasks that require absolute capability and have explicit budget approval.

### 3. Opus 4.8 — quality-critical / enterprise tier

API ID: `claude-opus-4-8`

Use for:
- Complex architecture, deep analysis, large refactors, enterprise or customer-facing output.
- Financially material, legally sensitive, compliance-sensitive, security-sensitive, or high-reputation work.
- Cases where Sonnet 5 at high effort is likely to underperform or has already failed.

Effort guidance:
- Start with `high` for most intelligence-sensitive work.
- Use `xhigh` for complex coding and agentic tasks that need extended exploration.
- Use `max` only when evals or business stakes justify the extra spend.

### 4. Fable 5 — frontier reserve

API ID: `claude-fable-5`

Use for:
- The hardest reasoning, novel system design, long-running agents, and failed Opus cases.
- Tasks where the value of a better answer materially exceeds the additional cost.

Never default to Fable. Route here only by explicit escalation, manual override, or an approved high-stakes policy.

## Manual override

A manually selected tier through a picker, `/model`, `tier=`, or environment override wins and remains locked for the current call/session. No automatic re-routing should override a manual choice except for access/availability fallback from Fable to a lower tier.

## Escalation rules

1. Start at Haiku only for mechanical tasks.
2. Start at Sonnet for normal production work.
3. Start at Opus for complex architecture, high-stakes analysis, customer-facing output, compliance, money-impacting work, or ambiguous quality-critical work.
4. Escalate one tier if output is too short, incomplete, tool-use fails, or confidence is low.
5. Escalate to Fable only for genuinely frontier tasks or after Opus fails.

### The 80%-then-ROI gate (mandatory before any tier escalation)

Escalating a model tier is never the first lever — raising effort within the current tier is. A tier
is not "used up" until it has been tried at the effort level the task actually calls for. Concretely,
before moving up a tier:

1. **Exhaust the current tier's effort range first.** If the current tier is still running at `low` or
   `medium` and the task is at all quality-sensitive, raise effort (`high` → `xhigh` → `max`) before
   changing model. A Sonnet-5-at-`xhigh` result is frequently as good as an Opus-4.8-at-`high` result
   at roughly a third of the per-token cost — jumping tiers first throws that away.
2. **Then escalate only when one of these is true:**
   - **STAKES override — skip the ROI check entirely.** Money-moving, legal/compliance, security,
     irreversible production actions, or anything else already gated `STAKES` in the deployment layer
     (`Claude-code-Agents` → `docs/model-routing-policy-v4.md`) escalates immediately regardless of
     cost. Never let a cost gate block a stakes-gated task.
   - **Verification failure at max effort.** The maxed-out attempt at the current tier fails a concrete
     check — tests don't pass, tool use errors out, output is empty/short/off-schema (the existing
     rule 4 guardrail), or a downstream reviewer/verifier flags it. This is evidence, not a guess —
     never escalate preemptively "just in case" when a cheap check hasn't actually found a problem.
   - **Marginal ROI is clearly positive.** No verification signal exists (subjective/creative/judgment
     work), but the task's stakes are high enough that a materially better answer is worth the
     multiplier below. "Materially better" means the escalation target has a real capability edge for
     *this specific task type* (see Domain guidance below) — not "it's more expensive so it must be
     better."
3. **Know the multiplier you're paying before you pay it.** Reference pricing (see Pricing reference
   below): Haiku → Sonnet is roughly a **3×** input/output cost jump; Sonnet → Opus is roughly **1.7×**;
   Opus → Fable is roughly **2×**. Compounding Haiku → Fable is on the order of **10×**. These are not
   hard caps — a stakes-gated task pays them without hesitation — but for the ROI-gated case, ask "is
   this task's marginal value materially higher than a 3×/1.7×/2× spend increase," not "would the
   fancier model probably do a bit better."

This gate applies to model-tier escalation only. It does not gate *effort* increases within a tier —
raising effort is the free first move, not something to ration.

### Grounded in published cascade-routing research (2026-07-08 review)

The shape above matches the published literature on LLM cascades, with three refinements worth
calling out explicitly:

1. **Our verification signal is a placeholder, not a calibrated threshold — say so.** `router.py`'s
   only automatic escalation trigger is "output is under 20 characters." Published cascade-routing
   work (e.g. *Cluster, Route, Escalate*, arXiv:2606.27457; *Is Escalation Worth It? A
   Decision-Theoretic Characterization of LLM Cascades*, arXiv:2605.06350) is explicit that real
   escalation thresholds should be set from evaluation data via a calibrated confidence signal, not a
   fixed heuristic — "quality has to be measured before it can be optimized." A 20-character cutoff
   catches truncation, nothing else. Known limitation, not a claim of correctness: until `router.py`
   logs a real quality/confidence signal per call, treat rule 4's guardrail as a crude floor, and lean
   more on human/test verification for anything that matters.
2. **Cascades win most cleanly on async/throughput work, not synchronous chat.** The same research
   notes cascades shine where "the blended-cost win outweighs the tail latency on escalations" —
   background agents, batch jobs, subagent fan-out. For latency-sensitive interactive turns, a failed
   low-tier attempt plus a retry costs the user visible wait time on top of the token cost; when in
   doubt on a synchronous, user-facing turn, it's reasonable to start one tier higher rather than
   count on the cascade to converge fast.
3. **Published cascades keep a large majority of traffic off the top tier.** One cited 2025 result
   (matrix-factorization LLM router, MT-Bench) reached ~95% of top-tier quality while sending only
   ~14% of queries to the strongest model. That's a useful sanity check, not a target to hit exactly:
   if a session's tier mix in practice skews heavily toward Opus/Fable, that's a signal the
   effort-first step of this gate isn't actually being applied, worth a spot-check rather than
   assuming the mix is simply "this session was hard."

This account's routing implements the recommended three-layer structure (deterministic STAKES regex
gate → cheap classifier pass → cascade-with-escalation) described in that literature:

1. **Deterministic STAKES regex gate** — `router.py`'s `looks_like_stakes()` (mirrored in
   `hq_orchestrator/core.py`) forces any customer money/legal task onto a Claude tier *before*
   classification, regardless of the caller's `stakes` flag. It is high-precision and lenient: a
   code review, a model valuation, or a pricing-*module* refactor is **not** stakes and runs on the
   bridge; only genuine customer commerce/legal (invoice, refund, GST, payment link, an actual
   `$amount`, contract terms) is caught. Add terms per-deployment with `CLAUDE_ROUTER_STAKES_KEYWORDS`.
2. **Cheap classifier pass** — the Haiku classifier picks a starting tier for non-stakes work.
3. **Cascade-with-escalation** — the one-strike incompleteness escalation (a crude 20-char/`max_tokens`
   floor, per the note above — not a calibrated confidence signal).

Note on the effort-first ("80%-then-ROI") gate above: that is **operating policy for the harness and
sessions** (the `/auto-escalate` skill applies it across model+effort dials), not something `router.py`
enforces per API call — `router.py` escalates by tier on the incompleteness signal. The two layers are
complementary, not the same mechanism.

## Domain guidance

The tier/effort tables above are task-class-based (mechanical / normal / quality-critical / frontier).
Layer domain on top — the right tier for a task also depends on *what kind* of work it is:

| Domain | Default tier | Effort | Notes |
|---|---|---|---|
| **Design** (UI/UX, visual, brand, layout) | Sonnet 5 | `high` | Escalate to Opus for a from-scratch design system or a client-facing brand deliverable; Sonnet at `high` handles iteration on an existing design language well. |
| **Code** (implementation, refactors, debugging) | Sonnet 5 | `high`, `xhigh` for hard agentic/coding tasks | Escalate to Opus for large multi-file architecture, security-sensitive code, or after a maxed-effort Sonnet attempt fails review/tests (see escalation gate above). |
| **Data & analysis** (extraction, transforms, reporting) | Haiku for pure mechanical extraction/reformatting; Sonnet for analysis requiring judgment | `high` on Sonnet | Never use Haiku when the task requires interpreting ambiguous data, only for deterministic transforms. |
| **Debugging** | Sonnet 5 at `xhigh` | — | Debugging is agentic and benefits disproportionately from higher effort before it benefits from a higher tier — push effort first per the gate above. |
| **Writing** (copy, docs, long-form) | Sonnet 5 | `high` | Escalate to Opus for enterprise-facing or reputationally sensitive copy (matches the existing quality-critical rule), not for routine drafting. |

## Pricing reference (self-updating — do not let this go stale)

Snapshot as of 2026-07-08, per 1M tokens (input / output):

| Tier | Model | Input | Output |
|---|---|---|---|
| Haiku | `claude-haiku-4-5` | $1.00 | $5.00 |
| Sonnet | `claude-sonnet-5` | $3.00 (**$2.00 introductory, through 2026-08-31**) | $15.00 (**$10.00 introductory**) |
| Opus | `claude-opus-4-8` | $5.00 | $25.00 |
| Fable | `claude-fable-5` | $10.00 | $50.00 |

`MODEL_REGISTRY` in `router.py` already carries these as `input_usd_per_mtok`/`output_usd_per_mtok` —
this table must stay in sync with that dict, not drift independently.

**This table has a known expiry.** Sonnet 5's introductory pricing ($2/$10) reverts to $3/$15 on
2026-08-31 — after that date the Haiku→Sonnet and Sonnet→Opus multipliers above both shift. Anyone
running this policy after that date should re-verify pricing before trusting the multiplier guidance,
via one of:
- `client.models.retrieve("claude-sonnet-5")` / `client.models.list()` (Models API — live, authoritative)
- The `claude-api` Claude Code skill's cached pricing table (re-synced periodically)

If either source disagrees with this table, the live source wins — update this table and
`router.py`'s registry together, don't patch just one.

## Effort rules

Supported effort values: `low`, `medium`, `high`, `xhigh`, `max`.

Default:
- Haiku: no effort setting.
- Sonnet 5: `high`.
- Opus 4.8: `high`.
- Fable 5: `high`.

Use lower effort only after evals prove quality holds. Use `xhigh` or `max` only when the task needs deeper reasoning and the budget is acceptable.

## Output standard

Produce the exact depth the task requires. No filler. Be concise for mechanical work and thorough for high-stakes work. For code changes, prefer runnable patches, tests, migration notes, and clear rollback instructions.

## Operational notes

- Keep model IDs configurable with environment variables: `CLAUDE_ROUTER_HAIKU_MODEL`, `CLAUDE_ROUTER_SONNET_MODEL`, `CLAUDE_ROUTER_OPUS_MODEL`, `CLAUDE_ROUTER_FABLE_MODEL`.
- Log every dispatch with timestamp, tier, model ID, effort, input tokens, output tokens, and status.
- Use the Models API periodically to confirm available models and token limits for the running account.
- Do not commit API keys, private transcripts, local paths, or session IDs into public repositories.

## Related — operational deployment policy

This document is the **implementation reference** for the router package
(`router.py` + tests: tiers `haiku`, `sonnet`, `glm`, `opus`, `fable` — since
v5.1 the Ollama-bridged `glm` tier is first-class in the package). The
**operational deployment layer** — standing auto-delegation, the
`UserPromptSubmit` classifier hook, and the HEAVY-offload conventions — is
documented separately in `edagher92-coder/Claude-code-Agents` →
`docs/model-routing-policy-v4.md` (v4.0 *Automatic Tier Delegation*). Both
layers share one bridge contract: `CLAUDE_ROUTER_OLLAMA_URL` / `OLLAMA_API_KEY`
mean the same thing everywhere.

## GLM 5.2 mid-tier (Ollama bridge) — v5.1 setup contract

`glm-5.2` sits between Sonnet and Opus for heavy NON-stakes bulk reasoning,
drafting, and summarising, dispatched through the Ollama bridge (both
`router.py` and `hq_orchestrator/ollama_caller.py`).

**Base resolution — a priority chain, first ready base wins:**
1. `CLAUDE_ROUTER_OLLAMA_URL` — one URL or a comma-separated list. Point it at
   a local daemon or a routing server on the tailnet (a daemon signed in to an
   Ollama account serves `:cloud` tags through the same endpoint). Defaults to
   `http://localhost:11434`.
2. `https://ollama.com` is appended automatically when `OLLAMA_API_KEY` is set
   (Ollama Cloud direct — the backstop). Reachability alone is not readiness:
   ollama.com answers probes unauthenticated but rejects generation without a
   key; `python router.py --doctor` flags exactly that.

**Model tag precedence:** `CLAUDE_ROUTER_GLM_MODEL` > `GLM_OLLAMA_TAG` >
registry default `glm-5.2:cloud`. Env is read at call time in both dispatch
paths, so long-lived servers pick up rotation without a restart.

**Degradation rules (all verified by `tests/test_router_v51_setup.py`):**
- Bridge unreachable + Anthropic available → re-route to `claude-sonnet-5`,
  never block.
- Anthropic unavailable (no SDK or no key) → OFFLINE mode: non-stakes work
  routes to the bridge, Claude tiers are skipped during escalation, and
  `stakes=True` refuses with `RouterSetupError` (NUMBERS RULE: customer-facing
  prices, quotes, invoices, and legal content never run on the bridge).
- Neither engine → `RouterSetupError` pointing at `--doctor`.

## Token efficiency (per-call, added 2026-07-20; Fable-reviewed)

Three levers that cut tokens **without cutting quality** — quality is the
ceiling, cost the floor. All opt-out via env, all covered by
`tests/test_router_v51_setup.py`.

1. **Output-efficiency directive** (`EFFICIENCY_DIRECTIVE`). Injected as the
   Anthropic `system` field and the Ollama `system` field on every real
   dispatch — never on the classifier call (5-token answer) or the compaction
   call. It bans preamble/filler/restating/padding and mandates dense form,
   while explicitly requiring that correctness and completeness outrank brevity
   (every number, step, caveat kept; end on a complete line so a terse answer
   isn't misread as truncated). Disable with `CLAUDE_ROUTER_CONCISE=0`.
2. **Input auto-compaction** (`_maybe_compact`). Oversized **non-stakes** input
   (est. tokens > `CLAUDE_ROUTER_COMPACT_THRESHOLD`, default 8000) is condensed
   on the cheapest engine before dispatch — but only when the target tier makes
   it pay (Sonnet/Opus/Fable; never Haiku-for-Haiku or flat-rate GLM unless near
   its window). A hard check (`_compaction_safe`) discards the compaction unless
   every number / URL / path / email from the original survives verbatim, so the
   NUMBERS RULE is enforced, not merely requested. Stakes tasks are **never**
   compacted; a truncated or failed compaction silently falls back to the
   original; on escalation the higher tier runs the **full original** text.
   Disable with `CLAUDE_ROUTER_COMPACT=0`.
3. **Widen-before-climb.** A token-cap truncation now gets one bigger-budget
   retry at the **same** tier (max_tokens ×4, capped at the tier's
   `max_output_tokens`) before escalating — so a long answer no longer truncates
   at every rung and climbs the whole ladder still truncated. Strictly raises
   completeness; costs nothing when answers fit.

`python router.py --doctor` shows the active efficiency settings. Flags:
`--no-concise`, `--no-compact`, `--compact-threshold N`.

**Complementary lever (session, not per-call):** Claude Code's own
`autoCompactEnabled` / `autoCompactWindow` in `settings.json` compacts a live
session's *context* before it inflates credit. That covers the running session;
the three levers above cover each per-call dispatch the router makes.
