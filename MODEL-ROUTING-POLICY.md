# Model Routing Policy — Quality First, Zero Waste

Principle: route every task to the model that guarantees a flawless
result. Never let cost degrade quality. Never burn a premium tier on
work a lower tier does perfectly.

1. Haiku 4.5 — MECHANICAL ONLY  [claude-haiku-4-5]
   Reformatting, renames, extraction, boilerplate, one-line answers.
   Any real reasoning → skip this tier.

2. Sonnet 5 — DEFAULT (~65-75% of tasks)  [claude-sonnet-5]
   General + agentic coding, drafting, business tasks, multi-step
   agent workflows, browser/terminal automation, long-context work
   (1M window). Use "high" effort before escalating.
   Avoid Extra High effort as a default — at xHigh, Sonnet 5 can cost
   more than Opus 4.8 at medium-high. If a task needs xHigh, route to
   Opus instead.

3. Opus 4.8 — QUALITY-CRITICAL (~20-30%)  [claude-opus-4-8]
   Maximum-accuracy work: complex architecture, deep analysis,
   nuanced writing, financially material or customer-facing output.
   When in doubt between Sonnet 5 high-effort and Opus, choose Opus.

4. Fable 5 — FRONTIER RESERVE (<5%)  [claude-fable-5]
   Hardest reasoning, novel system design, or when Opus output
   proved insufficient. Never default.

MANUAL OVERRIDE (highest priority): a manually selected model —
picker, /model, or tier= — wins and stays locked for the session.
No re-routing, no nagging. Routing resumes on new session or explicit
re-invocation.

Escalation: degraded output → bump Sonnet 5 effort (max High), then
re-run one tier up. Max one retry per tier. Suspended during manual
override.

Output: exact depth the task requires — no filler, full detail where
it matters.
