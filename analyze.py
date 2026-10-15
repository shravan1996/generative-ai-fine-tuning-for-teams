"""Turn the saved predictions into every number the article quotes.

    python analyze.py            # writes results/summary.json and prints it

Reads only files under results/ and data/splits/, so it runs without a
model, a GPU or a key.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from route import ROOT, SPLITS, meta, read_jsonl

RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"

DAY1 = ["haiku_day1", "sonnet_day1", "qwen_listed_day1", "qwen_k8_day1", "qwen_k32_day1", "qwen_all_day1"]
DAY90 = ["haiku_day90", "sonnet_day90", "qwen_all_day1_on_day90", "qwen_retrain_day90", "qwen_listed_day90"]


def load(arm):
    path = PREDS / f"{arm}.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    for n, r in enumerate(rows):
        r.setdefault("i", n)
    return {r["i"]: r for r in rows}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def accuracy(preds, keep=None):
    rows = [r for r in preds.values() if keep is None or keep(r)]
    k = sum(r["pred"] == r["label"] for r in rows)
    lo, hi = wilson(k, len(rows))
    return {"n": len(rows), "correct": k, "accuracy": round(k / len(rows), 4) if rows else None,
            "ci95": [round(lo, 4), round(hi, 4)],
            "no_valid_queue": sum(r["pred"] is None for r in rows)}


def mcnemar(a, b, keep=None):
    """Exact two-sided McNemar test on messages both arms answered."""
    ids = [i for i in a if i in b and (keep is None or keep(a[i]))]
    only_a = sum(a[i]["pred"] == a[i]["label"] and b[i]["pred"] != b[i]["label"] for i in ids)
    only_b = sum(b[i]["pred"] == b[i]["label"] and a[i]["pred"] != a[i]["label"] for i in ids)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n": len(ids), "only_first_right": only_a, "only_second_right": only_b, "p": round(p, 6)}


def auroc(pos, neg):
    """Probability a random positive scores above a random negative (ties count half)."""
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    rank_sum = 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + 1 + j) / 2
        rank_sum += r * sum(1 for t in allv[i:j] if t[1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    m = meta()
    new = set(m["new_queues"])
    is_new = lambda r: r["label"] in new
    is_old = lambda r: r["label"] not in new
    out = {"new_queues": m["new_queues"], "day1": {}, "day90": {}}

    # Day one: seventy queues.
    arms = {a: load(a) for a in DAY1}
    for a, p in arms.items():
        if p:
            out["day1"][a] = accuracy(p)
    if arms["haiku_day1"]:
        out["day1_vs_haiku"] = {a: mcnemar(arms[a], arms["haiku_day1"])
                                for a in DAY1[1:] if arms[a]}

    # Day ninety: seventy-seven queues, old and new messages apart.
    arms90 = {a: load(a) for a in DAY90}
    for a, p in arms90.items():
        if p:
            out["day90"][a] = {"old": accuracy(p, is_old), "new": accuracy(p, is_new)}

    # Does the day-one model's confidence flag the new queues?
    p = arms90["qwen_all_day1_on_day90"]
    if p:
        conf_new = [r["confidence"] for r in p.values() if is_new(r)]
        conf_old = [r["confidence"] for r in p.values() if is_old(r)]
        # Alarm score: low confidence. AUROC for "new" against "old".
        a = auroc([1 - c for c in conf_new], [1 - c for c in conf_old])
        cut = sorted(conf_old)[int(0.10 * len(conf_old))]
        caught = sum(c < cut for c in conf_new) / len(conf_new)
        dest = defaultdict(Counter)
        for r in p.values():
            if is_new(r):
                dest[r["label"]][r["pred"]] += 1
        out["alarm"] = {
            "auroc_new_vs_old": round(a, 4),
            "threshold_sending_10pct_old": cut,
            "new_caught_at_that_threshold": round(caught, 4),
            "median_conf_old": sorted(conf_old)[len(conf_old) // 2],
            "median_conf_new": sorted(conf_new)[len(conf_new) // 2],
            "share_new_at_or_above_0_9": round(sum(c >= 0.9 for c in conf_new) / len(conf_new), 4),
            "share_old_at_or_above_0_9": round(sum(c >= 0.9 for c in conf_old) / len(conf_old), 4),
            "where_new_messages_went": {q: dest[q].most_common(3) for q in sorted(dest)},
        }
        # The same question for Claude: does the 70-name prompt route new messages anywhere sensible?
    # Per new queue, for the arms that know it.
    for a in ("haiku_day90", "sonnet_day90", "qwen_retrain_day90"):
        p = arms90[a]
        if p:
            out.setdefault("new_queue_detail", {})[a] = {
                q: accuracy(p, lambda r, q=q: r["label"] == q)["accuracy"] for q in sorted(new)}

    # Batching check: Haiku one message per call against the same messages in batches.
    single, batched = load("haiku_day1_single"), arms["haiku_day1"]
    if single and batched:
        ids = [i for i in single if i in batched]
        out["batching_check"] = {
            "n": len(ids),
            "single_accuracy": round(sum(single[i]["pred"] == single[i]["label"] for i in ids) / len(ids), 4),
            "batched_accuracy": round(sum(batched[i]["pred"] == batched[i]["label"] for i in ids) / len(ids), 4),
            "agree": round(sum(single[i]["pred"] == batched[i]["pred"] for i in ids) / len(ids), 4),
        }

    # Cost and time.
    cost = {}
    for folder in sorted((RESULTS / "claude_raw").glob("*/usage.json")):
        u = json.loads(folder.read_text())
        if u["n"]:
            tokens_in = u["input"] + u["cache_write"] + u["cache_read"]
            cost[u["arm"]] = {
                "messages": u["n"], "calls": u["calls"],
                "input_tokens_per_message": round(tokens_in / u["n"], 1),
                "output_tokens_per_message": round(u["output"] / u["n"], 1),
                "list_usd_per_10k_messages": round(u["cost_usd_list"] / u["n"] * 10_000, 2),
            }
    out["claude_cost"] = cost
    runs = RESULTS / "local_runs.jsonl"
    if runs.exists():
        out["local_scoring"] = {r["arm"]: {"n": r["n"], "messages_per_second": round(r["n"] / r["wall_seconds"], 2)}
                                for r in read_jsonl(runs)}
    train = {}
    # Adapters stay out of git; their run records are copied in beside the results.
    (RESULTS / "training").mkdir(parents=True, exist_ok=True)
    for run in sorted((ROOT / "adapters").glob("*/run.json")):
        (RESULTS / "training" / f"{run.parent.name}.json").write_text(run.read_text())
    for run in sorted((RESULTS / "training").glob("*.json")):
        r = json.loads(run.read_text())
        train[run.stem] = {k: r[k] for k in ("rows", "updates", "train_seconds", "peak_memory_gb",
                                              "trainable_parameters")}
    out["training"] = train

    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
