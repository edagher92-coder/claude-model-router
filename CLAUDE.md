# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## Repository purpose

`claude-model-router` is a small, standalone Python package that routes API
tasks across the current Claude model lineup — Haiku 4.5, Sonnet 5, Opus 4.8,
Fable 5, and a manual-only Mythos 5 — with cost control, effort settings,
usage logging, and safe fallback behavior. It is public/free-kit-simple: one
module (`router.py`), one policy doc (`MODEL-ROUTING-POLICY.md`), one test
file, no framework dependencies beyond the `anthropic` SDK.

**This repo is the canonical source of truth for the account's model-routing
policy.** `Claude-code-Agents/CLAUDE.md` (sibling repo) states explicitly that
its `docs/model-routing-policy-v4.md` treats `edagher92-coder/claude-model-router`
as the implementation source of truth — meaning other repos in the
`edagher92-coder` account vendor or reference this repo's policy rather than
maintaining their own copy. Fix routing logic or policy wording here first,
then propagate to any repo that copies it.

## How the pieces relate

- **`MODEL-ROUTING-POLICY.md`** is the human-readable policy: which tier to
  start at, when to escalate, default effort per tier, and the rule that a
  manual model/tier selection always wins over automatic classification. It
  is the spec.
- **`router.py`** is the implementation of that spec:
  - `MODEL_REGISTRY` — a `ModelInfo` dataclass per tier (`haiku`, `sonnet`,
    `opus`, `fable`, `mythos`) with API ID, role description, context window,
    max output tokens, per-million-token pricing, and whether the model
    supports an `effort` setting.
  - `classify(task)` / `_classify` — asks Haiku to label a task HAIKU / SONNET
    / OPUS / FABLE (never MYTHOS) and maps that to a tier, defaulting to
    `sonnet` on an unrecognised reply.
  - `run(task, tier=None, effort=None, max_tokens=4096)` — the main entry
    point. Uses the manual `tier` if given, otherwise classifies the task;
    dispatches to `client.messages.create(...)`, passing `output_config={"effort": ...}`
    for tiers that support it (all but Haiku).
  - `ESCALATE` / `FALLBACK` dicts — `ESCALATE` bumps a tier up one rung
    (`haiku→sonnet→opus→fable`) when a response is suspiciously short
    (`< 20` stripped characters) and the tier isn't already `fable`;
    `FALLBACK` steps a tier down (`mythos→fable→opus→sonnet→haiku`) when a
    Fable/Mythos call errors with an access/availability status
    (`400/401/403/404`), retrying up to 4 attempts total.
  - Mythos is disabled by default and only becomes selectable when
    `CLAUDE_ROUTER_ENABLE_MYTHOS=true` is set (`_tier_enabled`); the
    classifier is instructed never to choose it.
  - Every dispatch is appended to a CSV log (`router-usage.csv` by default,
    overridable via `CLAUDE_ROUTER_LOG`) with timestamp, tier, model ID,
    effort, input/output tokens, and status — including `fallback_<code>`
    rows when a tier falls back.
  - Per-tier model IDs and effort defaults can be overridden via environment
    variables (`CLAUDE_ROUTER_<TIER>_MODEL`, `CLAUDE_ROUTER_<TIER>_EFFORT`,
    and a global `CLAUDE_ROUTER_EFFORT` fallback), so the same code can point
    at account-specific model aliases without edits.
  - `registry()` returns the live registry (including env overrides and the
    resolved `enabled` flag) for introspection.

In short: `MODEL-ROUTING-POLICY.md` says what the routing rules *should* be;
`router.py` is the executable version of those same rules, tested against
`tests/test_router_registry.py`.

## Local validation

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/test_router_registry.py` covers: the registry contains exactly the
five current tiers with the expected API IDs; environment variable overrides
of a tier's model ID take effect; Mythos stays disabled until
`CLAUDE_ROUTER_ENABLE_MYTHOS=true` is set; `_effort_for` returns `None` for
Haiku (no effort support) and the expected value for Sonnet/Opus; and invalid
tier/effort inputs raise `ValueError` with the expected message. Tests
monkeypatch `ANTHROPIC_API_KEY` and reload the module — no live API calls or
network access required.

There is no linter/formatter configured for this repo yet.

## Branch & PR convention

- Feature work happens on a `claude/<short-slug>` branch (e.g. the current
  branch, `claude/md-folder-reorganize-d9m343`).
- Push and open a PR against `main` for review; do not push directly to `main`.
- Run `pytest` before pushing.

## Safety notes (from README, still true)

- Never commit API keys, private transcripts, local machine paths, session
  IDs, or customer data — this includes `router-usage.csv`, which is
  generated locally at runtime and should not be committed if it contains
  real usage data.
- Manual tier/model overrides always win over automatic classification.
- Mythos 5 is manual/approved-access only and must stay disabled by default.

## What NOT to do

- Don't invent new tiers, models, or CLI flags that aren't in `router.py` —
  this file should only describe verified, current behavior.
- Don't auto-route to Mythos or change its default-disabled behavior.
- Don't remove the fallback/escalation guardrails (`ESCALATE`, `FALLBACK`,
  the short-response retry, the Fable/Mythos access-error fallback) without
  updating both `MODEL-ROUTING-POLICY.md` and the tests in the same change,
  since sibling repos treat this repo's behavior as the reference
  implementation.
