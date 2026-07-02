# claude-model-router

**Cut your Claude API/Max burn 30–45% without sacrificing quality.**

A tiny Haiku classifier reads each task (one API call, ~5 tokens, < $0.001) and dispatches it to the cheapest model that guarantees a flawless result. Every call is logged so you can prove the savings.

## Quick start

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python router.py
```

## What’s in the free kit (this repo)

- `router.py` — classifier + dispatch + usage log (MIT licensed)
- `MODEL-ROUTING-POLICY.md` — the quality-first routing policy, ready to paste into any CLAUDE.md
- `LICENSE` — MIT

## Want the full Pro Kit?

The Pro Kit adds:
- `install.sh` / `install.ps1` — one command: policy into ~/.claude/CLAUDE.md, default model set, router dropped into your repos
- `CLAUDE.md` — drop-in for Claude Code / Cowork sessions
- A UserPromptSubmit hook (zero API calls, ~ to bypass)
- `usage-dashboard.html` — drag your CSV in, see tier split + spend estimate
- Lifetime updates for every new Claude model
- Email support from the builder

Want the one-command installer, Claude Code hook, usage dashboard, and priority support? → **[Pro Kit — link here]**

MIT licensed. Built by [Elie Dagher](https://github.com/) — Snowflow NSW, Slushieco, ReGen Labs Engineering, DISPATCHIQ.