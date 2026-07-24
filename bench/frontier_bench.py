#!/usr/bin/env python3
"""frontier_bench — head-to-head frontier-model bench on OUR probes.

Runs the exact probe set from model_bench.py (extract / summarise / code /
reason / price-honesty / tier-math / deep-reason) against frontier APIs so
vendor leaderboards can be cross-checked on the workload that actually
matters here. First matchup: Claude Opus 5 vs GPT-5.6 Sol.

Zero SDKs — plain urllib against both HTTP APIs. Keys come from env:
    ANTHROPIC_API_KEY   -> benches the Anthropic entrant(s)
    OPENAI_API_KEY      -> benches the OpenAI entrant(s)
A missing key skips that vendor with a clear note (never a crash), so the
report always says exactly what ran and what was skipped.

Usage:
    python3 bench/frontier_bench.py
    python3 bench/frontier_bench.py --anthropic-models claude-opus-5,claude-sonnet-5 \
                                    --openai-models gpt-5.6-sol
Writes bench/reports/frontier-YYYY-MM-DD.{json,md}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from model_bench import probes  # identical probe set — that's the whole point

REPORTS = HERE / "reports"
TIMEOUT = 240

# $/Mtok (in, out) for the cost-per-run column. Update alongside price changes.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-fable-5": (10.0, 50.0),
    "gpt-5.6-sol": (5.0, 30.0),
    "gpt-5.6-terra": (2.5, 15.0),
    "gpt-5.6-luna": (1.0, 6.0),
}


def _post(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def call_anthropic(model: str, prompt: str, key: str) -> tuple[str, int, int]:
    body = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
        {"model": model, "max_tokens": 1024,
         "messages": [{"role": "user", "content": prompt}]},
    )
    text = "".join(b.get("text", "") for b in body.get("content", []))
    usage = body.get("usage", {})
    return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def call_openai(model: str, prompt: str, key: str) -> tuple[str, int, int]:
    # max_completion_tokens (not max_tokens) — required by reasoning-tier models;
    # generous cap so hidden reasoning can't starve the visible answer.
    body = _post(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {key}"},
        {"model": model, "max_completion_tokens": 4096,
         "messages": [{"role": "user", "content": prompt}]},
    )
    text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = body.get("usage", {})
    return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def _openai_candidates(key: str) -> list[str]:
    """On a model-not-found, list what the account CAN see so the next run can
    be pointed at the real ID instead of a guess."""
    try:
        req = urllib.request.Request("https://api.openai.com/v1/models",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            models = [m.get("id", "") for m in json.loads(r.read().decode()).get("data", [])]
        return sorted(m for m in models if "5.6" in m or "sol" in m.lower())[:20]
    except Exception:
        return []


def bench_model(vendor: str, model: str, caller, key: str) -> dict:
    rows: dict = {}
    for name, (prompt, check) in probes().items():
        started = time.time()
        try:
            text, in_tok, out_tok = caller(model, prompt, key)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:200]
            except Exception:
                pass
            rows[name] = {"pass": False, "error": f"HTTP {e.code}: {detail}"}
            if e.code == 404 and vendor == "openai":
                rows["_model_candidates"] = _openai_candidates(key)
                break  # wrong model id — no point burning the remaining probes
            continue
        except Exception as e:  # noqa: BLE001 — record and move on
            rows[name] = {"pass": False, "error": str(e)[:200]}
            continue
        rows[name] = {
            "pass": bool(check(text)),
            "latency_s": round(time.time() - started, 2),
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "reply_head": text.strip()[:120],
        }
    return rows


def summarise(model: str, rows: dict) -> dict:
    probes_run = {k: v for k, v in rows.items() if not k.startswith("_") and "latency_s" in v}
    errors = {k: v for k, v in rows.items() if not k.startswith("_") and "error" in v}
    passes = sum(1 for v in probes_run.values() if v["pass"])
    total = len(probes_run) + len(errors)
    lats = [v["latency_s"] for v in probes_run.values()]
    in_tok = sum(v["in_tokens"] for v in probes_run.values())
    out_tok = sum(v["out_tokens"] for v in probes_run.values())
    pin, pout = PRICES.get(model, (0.0, 0.0))
    cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return {
        "passes": passes, "total": total,
        "avg_latency_s": round(sum(lats) / len(lats), 2) if lats else None,
        "in_tokens": in_tok, "out_tokens": out_tok,
        "run_cost_usd": round(cost, 4),
        "clean_sweep": passes == total and total == len(probes()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anthropic-models", default="claude-opus-5")
    p.add_argument("--openai-models", default="gpt-5.6-sol")
    a = p.parse_args()

    akey = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    okey = os.environ.get("OPENAI_API_KEY", "").strip()

    entrants: list[tuple[str, str]] = []
    skipped: list[str] = []
    for m in [m.strip() for m in a.anthropic_models.split(",") if m.strip()]:
        (entrants.append(("anthropic", m)) if akey else skipped.append(f"{m} (ANTHROPIC_API_KEY unset)"))
    for m in [m.strip() for m in a.openai_models.split(",") if m.strip()]:
        (entrants.append(("openai", m)) if okey else skipped.append(f"{m} (OPENAI_API_KEY unset)"))

    if not entrants:
        print("No API keys set — nothing to bench. Add ANTHROPIC_API_KEY and/or "
              "OPENAI_API_KEY (repo secrets for the workflow, env locally).")
        return 0

    date = dt.date.today().isoformat()
    results: dict = {"date": date, "probes": list(probes().keys()), "models": {},
                     "skipped": skipped}
    for vendor, model in entrants:
        print(f"── {model} ({vendor})")
        caller = call_anthropic if vendor == "anthropic" else call_openai
        key = akey if vendor == "anthropic" else okey
        rows = bench_model(vendor, model, caller, key)
        rows["_summary"] = summarise(model, rows)
        results["models"][model] = rows
        s = rows["_summary"]
        print(f"   {s['passes']}/{s['total']} pass, avg {s['avg_latency_s']}s, "
              f"~${s['run_cost_usd']} for the run")
        for k, v in rows.items():
            if not k.startswith("_") and "error" in v:
                print(f"   ! {k}: {v['error']}")
        if rows.get("_model_candidates"):
            print(f"   model id not found — account sees: {rows['_model_candidates']}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"frontier-{date}.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = [f"# Frontier head-to-head — {date}", "",
             "Same 7 probes as the weekly fleet bench (incl. price-honesty, "
             "tier-math, deep-reason).", "",
             "| Model | Pass | Avg latency | Out tokens | Run cost |", "|---|---|---|---|---|"]
    for model, rows in results["models"].items():
        s = rows["_summary"]
        lines.append(f"| {model} | {s['passes']}/{s['total']} | {s['avg_latency_s']}s "
                     f"| {s['out_tokens']} | ${s['run_cost_usd']} |")
    for note in skipped:
        lines.append(f"\n_Skipped: {note}_")
    (REPORTS / f"frontier-{date}.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {REPORTS / f'frontier-{date}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
