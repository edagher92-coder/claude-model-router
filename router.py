"""Model router v3.1 — Haiku classifier + dispatch + usage log.
Enforces MODEL-ROUTING-POLICY-v3.1.md mechanically for API traffic.
Usage:  from router import run;  reply = run("your task here")
Requires: pip install anthropic ; ANTHROPIC_API_KEY in env.
Logs every dispatch to router-usage.csv (timestamp, tier, in/out tokens).
"""
import csv
import datetime
import pathlib

import anthropic

client = anthropic.Anthropic()

LOG = pathlib.Path(__file__).parent / "router-usage.csv"

TIERS = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-5",
    "opus": "claude-opus-4-8",
    "fable": "claude-fable-5",
}
ESCALATE = {"haiku": "sonnet", "sonnet": "opus", "opus": "fable", "fable": "fable"}

CLASSIFIER_PROMPT = """You are a routing classifier. Reply with exactly one word.

HAIKU  — mechanical only: reformat, rename, extract, boilerplate, one-line answers.
SONNET — default: general coding, drafting, business tasks, multi-step agent work, browser/terminal automation, long-context (1M window).
OPUS   — quality-critical: complex architecture, deep analysis, nuanced writing, financially material or customer-facing output.
FABLE  — frontier reserve (<5%): hardest reasoning, novel system design, or when Opus output proved insufficient.

Task: {task}"""

def _classify(task: str) -> str:
    msg = client.messages.create(
        model=TIERS["haiku"],
        max_tokens=5,
        messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(task=task)}],
    )
    word = msg.content[0].text.strip().upper()
    return {"HAIKU": "haiku", "SONNET": "sonnet", "OPUS": "opus", "FABLE": "fable"}.get(word, "sonnet")

def _log(tier: str, in_tok: int, out_tok: int) -> None:
    with open(LOG, "a", newline="") as f:
        csv.writer(f).writerow([datetime.datetime.utcnow().isoformat(), tier, in_tok, out_tok])

def run(task: str, max_tokens: int = 4096) -> str:
    tier = _classify(task)
    for attempt in range(4):
        model = TIERS[tier]
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": task}],
        )
        _log(tier, resp.usage.input_tokens, resp.usage.output_tokens)
        text = resp.content[0].text

        if len(text.strip()) < 20 and tier != "fable":
            tier = ESCALATE[tier]
            continue
        return text
    return text

if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) or "Hello! What can you do?"
    print(run(task))