"""Route held-out messages with Claude, prompted with the queue list.

    python score_claude.py --arm haiku_day1 --model claude-haiku-5-5 --effort low \
        --test test_old.jsonl --day 1

Calls go through the Claude Code CLI in print mode (`claude -p`), with
tools, settings and MCP servers switched off, 20 messages per call
unless --batch-size says otherwise. Every
raw reply is saved under results/claude_raw/, so a rerun only calls for
batches that are missing, and the scorer works offline from the files.
"""

import argparse
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from route import ROOT, SPLITS, normalise, queues, read_jsonl

RAW = ROOT / "results" / "claude_raw"
PREDS = ROOT / "results" / "preds"
DEFAULT_BATCH = 20
SHUFFLE_SEED = 7

SYSTEM = "You route bank customer messages to queues for a contact centre. Answer with JSON only."
USER = (
    "Queues:\n{queues}\n\n"
    "Route each message below to exactly one queue from the list. Reply with a JSON array of "
    "{n} queue names, one per message, in the same order, and nothing else.\n\n{messages}"
)


def batches(rows, size):
    order = list(range(len(rows)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    return [order[i:i + size] for i in range(0, len(order), size)]


def call(model, effort, prompt):
    cmd = ["claude", "-p", "--model", model, "--system-prompt", SYSTEM, "--tools", "",
           "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
           "--output-format", "json"]
    if effort:
        cmd += ["--effort", effort]
    out = subprocess.run(cmd + [prompt], capture_output=True, text=True, timeout=300)
    return json.loads(out.stdout)


MAX_RETRIES = 2


def arrays(text):
    """Every JSON array of strings in a reply, in the order they appear."""
    dec, found = json.JSONDecoder(), []
    i = text.find("[")
    while i != -1:
        try:
            value, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("[", i + 1)
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            found.append(value)
        i = text.find("[", end)
    return found


def parse(result, n):
    """The last list of exactly n queue names in a reply. Sonnet sometimes
    prints a list, writes "Wait, I need to correct that", and prints a
    second one; the second is its answer. Failing that, the last list,
    padded or cut to n."""
    found = arrays(result or "")
    exact = [a for a in found if len(a) == n]
    if exact:
        return exact[-1]
    if found:
        return (found[-1] + [None] * n)[:n]
    return [None] * n


def complete(result, n):
    return any(len(a) == n for a in arrays(result or ""))


def attempt_paths(folder, i):
    return [folder / f"batch_{i:03d}.json"] + [folder / f"batch_{i:03d}_retry{k}.json"
                                               for k in range(1, MAX_RETRIES + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--test", required=True)
    ap.add_argument("--day", type=int, required=True, choices=[1, 90])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--limit-batches", type=int, default=None)
    args = ap.parse_args()

    allowed = queues(args.day)
    rows = read_jsonl(SPLITS / args.test)
    plan = batches(rows, args.batch_size)[: args.limit_batches]
    folder = RAW / args.arm
    folder.mkdir(parents=True, exist_ok=True)

    def ask(i):
        idx = plan[i]
        messages = "\n".join(f"{k + 1}. {json.dumps(rows[j]['text'], ensure_ascii=False)}"
                             for k, j in enumerate(idx))
        prompt = USER.format(queues="\n".join(allowed), n=len(idx), messages=messages)
        try:
            response = call(args.model, args.effort, prompt)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            print(f"batch {i}: {type(e).__name__}", flush=True)
            return None
        return {"indices": idx, "model": args.model, "effort": args.effort, "response": response}

    def run(i):
        # One call per batch, plus up to two retries when a reply does not
        # hold one queue name per message. Every reply is kept.
        n = len(plan[i])
        for k, path in enumerate(attempt_paths(folder, i)):
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["response"].get("is_error"):
                    saved = None
            else:
                saved = None
            if saved is None:
                saved = ask(i)
                if saved is None:
                    return
                path.write_text(json.dumps(saved, ensure_ascii=False) + "\n")
                print(f"{args.arm}: batch {i + 1}/{len(plan)}" + (f" retry {k}" if k else ""), flush=True)
            if complete(saved["response"].get("result", ""), n):
                return

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, range(len(plan))))

    # Score from the saved files only.
    allowed_set = set(allowed)
    preds, usage = [], {"calls": 0, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0,
                        "cost_usd_list": 0.0, "api_ms": 0}
    usage["retried_batches"] = 0
    usage["batches_never_complete"] = 0
    for i in range(len(plan)):
        tries = [json.loads(p.read_text()) for p in attempt_paths(folder, i) if p.exists()]
        if not tries:
            continue
        for saved in tries:
            r = saved["response"]
            u = r.get("usage", {})
            usage["calls"] += 1
            usage["input"] += u.get("input_tokens", 0)
            usage["cache_write"] += u.get("cache_creation_input_tokens", 0)
            usage["cache_read"] += u.get("cache_read_input_tokens", 0)
            usage["output"] += u.get("output_tokens", 0)
            usage["cost_usd_list"] += r.get("total_cost_usd", 0.0)
            usage["api_ms"] += r.get("duration_api_ms", 0)
        n = len(tries[0]["indices"])
        usage["retried_batches"] += len(tries) > 1
        good = [t for t in tries if complete(t["response"].get("result", ""), n)]
        usage["batches_never_complete"] += not good
        chosen = good[0] if good else tries[-1]
        answers = parse(chosen["response"].get("result", ""), n)
        for j, raw in zip(chosen["indices"], answers):
            preds.append({"i": j, "text": rows[j]["text"], "label": rows[j]["label"], "raw": raw,
                          "pred": normalise(raw, allowed_set)})
    PREDS.mkdir(parents=True, exist_ok=True)
    preds.sort(key=lambda p: p["i"])
    with open(PREDS / f"{args.arm}.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    acc = sum(p["pred"] == p["label"] for p in preds) / max(1, len(preds))
    usage["cost_usd_list"] = round(usage["cost_usd_list"], 4)
    (folder / "usage.json").write_text(json.dumps({"arm": args.arm, "n": len(preds), **usage}, indent=2) + "\n")
    print(f"{args.arm}: accuracy {acc:.3f} on {len(preds)} messages, {usage}")


if __name__ == "__main__":
    main()
