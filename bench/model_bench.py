"""Weekly Ollama model bench for router v5.1 delegation tuning.

Probes each candidate model on the task families the router actually delegates
(mechanical extract, bulk summarise, code, short reasoning) plus a
price-honesty probe (does the model invent a Snow Flow price, or say UNKNOWN?)
— the deciding factor for what non-stakes work it may touch.

Zero deps (stdlib). Usage:
    OLLAMA_API_KEY=...  python bench/model_bench.py                # default shortlist
    python bench/model_bench.py --models glm-5.2,gpt-oss:120b      # explicit
    python bench/model_bench.py --base http://localhost:11434      # your daemon

Writes bench/reports/<date>.md + .json. Run weekly (or via the
model-bench.yml workflow once OLLAMA_API_KEY is a repo secret).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
DEFAULT_MODELS = [
    "glm-5.2", "gpt-oss:120b", "qwen3.5:397b", "kimi-k2.7-code",
    "nemotron-3-nano:30b", "deepseek-v4-pro", "mistral-large-3:675b",
]
# Claude baselines run head-to-head on the identical probes whenever
# ANTHROPIC_API_KEY is present (skipped cleanly otherwise). This is what turns
# "published benchmarks say Sonnet-class" into a measured, same-task answer.
DEFAULT_BASELINES = ["claude-sonnet-5"]
MAX_TOKENS = 300
TIMEOUT = 180

SUMMARY_TEXT = (
    "Commercial slushy machines need a nightly strip-clean during trading periods. "
    "Syrup left in the bowl overnight thickens, blocks the tap and strains the auger "
    "motor, which is the single most common cause of summer breakdowns. A weekly "
    "deep-clean of the bowl seals prevents leaks. Operators who follow the schedule "
    "see far fewer callouts in December and January, when service techs are booked "
    "out weeks ahead and a dead machine means lost trade at the worst possible time."
)


def probes() -> dict:
    """probe name -> (prompt, checker(text) -> bool)."""
    def check_extract(t):
        return "kim@venue.com.au" in t and "ops@slushfest.com" in t and len(t) < 200

    def check_summary(t):
        sentences = [s for s in t.replace("\n", " ").split(".") if s.strip()]
        return 1 <= len(sentences) <= 3 and ("clean" in t.lower() or "breakdown" in t.lower())

    def check_code(t):
        # Model output is untrusted — never exec it in this process (restricted
        # globals are escapable via type-object chains). Run it in an isolated
        # subprocess: -I (isolated mode), cleared env, 10s timeout, and assert
        # on printed test results only.
        import subprocess
        code = t
        if "```" in code:  # strip a fence if present
            parts = code.split("```")
            code = max(parts, key=lambda p: "def " in p)
            code = code.replace("python", "", 1) if code.lstrip().startswith("python") else code
        harness = code + (
            "\n\nprint('BENCH_OK' if ("
            "is_palindrome('A man, a plan, a canal: Panama') "
            "and not is_palindrome('slushy') and is_palindrome('')) else 'BENCH_FAIL')\n"
        )
        try:
            proc = subprocess.run(
                ["python3", "-I", "-c", harness],
                capture_output=True, text=True, timeout=10, env={},
            )
            return "BENCH_OK" in proc.stdout
        except Exception:
            return False

    def check_reason(t):
        low = t.lower()
        return "carol" in low and "alice" not in low.replace("alice is", "")

    def check_honesty(t):
        low = t.lower()
        invented_price = "$" in t and any(c.isdigit() for c in t)
        return ("unknown" in low or "don't know" in low or "not sure" in low or "cannot" in low) and not invented_price

    def check_tier_math(t):
        import re as _re
        m = _re.search(r"-?\d+", t)
        return bool(m and m.group(0) == "2")

    return {
        "extract": (
            "Return ONLY the email addresses found in this text, one per line, nothing else:\n"
            "'Kim (kim@venue.com.au) asked about a hire; loop in ops@slushfest.com re the festival.'",
            check_extract,
        ),
        "summarise": (
            f"Summarise the following in exactly 2 sentences:\n\n{SUMMARY_TEXT}",
            check_summary,
        ),
        "code": (
            "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric "
            "characters, returns True for the empty string. Return ONLY the code, no explanation.",
            check_code,
        ),
        "reason": (
            "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? "
            "Answer with just the name.",
            check_reason,
        ),
        "price-honesty": (
            "What does Snow Flow Sydney charge to hire a double bowl slushy machine? "
            "If you do not have verified pricing, reply with the single word UNKNOWN.",
            check_honesty,
        ),
        "tier-math": (
            # Harder, business-shaped: requires the round-UP-to-the-next-tier
            # logic (the overquote rule). 460 needed, base 240, add-ons of 120:
            # 1 add-on = 360 (short), so 2. Models that round down say 1.
            "A hire must cover 460 serves. The base package covers 240 serves; "
            "extra capacity comes ONLY in whole add-ons of 120 serves each, and "
            "you must never provide fewer serves than required. How many add-ons? "
            "Answer with just the number.",
            check_tier_math,
        ),
    }


def generate_anthropic(model: str, prompt: str) -> tuple[str, float, int]:
    """Claude baseline call on the identical probe. Requires the anthropic
    package + ANTHROPIC_API_KEY; callers skip baselines when unavailable."""
    import anthropic
    client = anthropic.Anthropic()
    start = time.time()
    response = client.messages.create(
        model=model, max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start
    text = "\n".join(b.text for b in response.content if getattr(b, "text", "")).strip()
    return text, latency, int(getattr(response.usage, "output_tokens", 0))


def generate(base: str, api_key: str, model: str, prompt: str) -> tuple[str, float, int]:
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        # think:false — thinking models otherwise burn the whole num_predict
        # budget on hidden reasoning and return an empty `response` (observed
        # live with glm-5.2 and qwen3.5 on 2026-07-17). Ignored by non-thinkers.
        "think": False,
        "options": {"num_predict": MAX_TOKENS},
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(base + "/api/generate", data=payload, headers=headers)
    start = time.time()
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    latency = time.time() - start
    return (body.get("response") or "").strip(), latency, int(body.get("eval_count") or 0)


def discover_models(base: str, api_key: str) -> list:
    """Model names the bridge advertises via /api/tags. Lets the weekly bench
    auto-include a newly-released cloud model (e.g. Kimi K3 Max) with no code
    change and no tag guessing. Best-effort: returns [] on any error."""
    import urllib.request
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    try:
        req = urllib.request.Request(base + "/api/tags", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8"))
        return [m.get("name") for m in body.get("models", []) if m.get("name")]
    except Exception as exc:  # noqa: BLE001 - discovery is optional, never fatal
        print(f"discover: /api/tags failed ({exc}); using the static model list", flush=True)
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="router v5.1 weekly model bench")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES),
                        help="Claude models benched head-to-head on the same probes "
                             "(skipped unless ANTHROPIC_API_KEY is set); '' disables")
    parser.add_argument("--base", default="")
    parser.add_argument("--out-dir", default=str(HERE / "reports"))
    parser.add_argument("--discover", action="store_true",
                        help="also bench every model the bridge lists in /api/tags "
                             "(auto-picks up NEW cloud models like Kimi K3 Max the day "
                             "they land — no tag guessing, runs PC-off in CI)")
    args = parser.parse_args()

    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    base = (args.base or ("https://ollama.com" if api_key else "http://localhost:11434")).rstrip("/")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.discover:
        found = discover_models(base, api_key)
        added = [m for m in found if m not in models]
        if added:
            print(f"discover: +{len(added)} new model(s): {', '.join(added)}", flush=True)
        models = list(dict.fromkeys(models + found))
    baselines = [m.strip() for m in args.baselines.split(",") if m.strip()]
    if baselines and not os.getenv("ANTHROPIC_API_KEY", "").strip():
        print("note: ANTHROPIC_API_KEY unset — Claude baselines skipped", flush=True)
        baselines = []
    today = dt.date.today().isoformat()

    results: dict = {"date": today, "base": base, "models": {}}
    for model in models + baselines:
        is_baseline = model in baselines
        row: dict = {"baseline": is_baseline} if is_baseline else {}
        for name, (prompt, check) in probes().items():
            try:
                if is_baseline:
                    text, latency, tokens = generate_anthropic(model, prompt)
                else:
                    text, latency, tokens = generate(base, api_key, model, prompt)
                row[name] = {"pass": bool(check(text)), "latency_s": round(latency, 1),
                             "tokens": tokens, "reply_head": text[:120]}
            except Exception as exc:  # noqa: BLE001 - a dead model must not kill the bench
                row[name] = {"pass": False, "error": str(exc)[:200]}
            print(f"{model:24s} {name:14s} "
                  f"{'PASS' if row[name].get('pass') else 'FAIL':4s} "
                  f"{row[name].get('latency_s', '-')}s", flush=True)
        results["models"][model] = row

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Same-day reruns MERGE per-model (fresh rows win) instead of clobbering
    # the whole report — a partial re-test must not erase the full table.
    json_path = out_dir / f"{today}.json"
    if json_path.exists():
        try:
            prior = json.loads(json_path.read_text(encoding="utf-8"))
            merged = prior.get("models", {})
            merged.update(results["models"])
            results["models"] = merged
        except (ValueError, OSError):
            pass  # unreadable prior report: overwrite it
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [f"# Model bench — {today}", "", f"Base: `{base}` · max_tokens={MAX_TOKENS}", "",
             "| model | " + " | ".join(probes()) + " | avg latency |", "|---|" + "---|" * (len(probes()) + 1)]
    for model, row in results["models"].items():
        cells, lats = [], []
        for name in probes():
            r = row.get(name)
            if r is None:  # merged older row from before a probe existed
                cells.append("—")
                continue
            cells.append(("PASS" if r.get("pass") else "FAIL") + (f" {r['latency_s']}s" if "latency_s" in r else " (err)"))
            if "latency_s" in r:
                lats.append(r["latency_s"])
        avg = f"{sum(lats) / len(lats):.1f}s" if lats else "-"
        label = f"**{model}** (baseline)" if row.get("baseline") else model
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {avg} |")
    (out_dir / f"{today}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {out_dir / (today + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
