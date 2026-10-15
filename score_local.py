"""Route held-out messages with the small model, prompted or fine-tuned.

    python score_local.py --arm qwen_listed_day1 --test test_old.jsonl --day 1 --listed
    python score_local.py --arm qwen_k32_day1 --test test_old.jsonl --day 1 --adapter adapters/day1_k32

Greedy decoding, one message at a time. The confidence recorded for
each answer is the probability the model gave its own reply, end token
included, which is the number a team would threshold to send unsure
messages to a person.
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from route import LISTED_PROMPT, ROOT, SPLITS, TRAINED_PROMPT, normalise, queues, read_jsonl

MODEL = "Qwen/Qwen3.5-0.8B"
PREDS = ROOT / "results" / "preds"


def greedy(model, eos, prompt_ids, max_new=24):
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)[:, -1, :]
    out, logp = [], 0.0
    for _ in range(max_new):
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        t = mx.argmax(lp, axis=-1)
        ti = t.item()
        logp += lp[0, ti].item()
        if ti in eos:
            break
        out.append(ti)
        logits = model(t.reshape(1, 1), cache=cache)[:, -1, :]
    return out, logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--day", type=int, required=True, choices=[1, 90])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--listed", action="store_true", help="put the queue list in the prompt")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    allowed = queues(args.day)
    allowed_set = set(allowed)
    model, tok = load(MODEL, adapter_path=args.adapter)
    eos = set(tok.eos_token_ids) | {tok.convert_tokens_to_ids("<|im_end|>")}

    rows = read_jsonl(SPLITS / args.test)[: args.limit]
    PREDS.mkdir(parents=True, exist_ok=True)
    path = PREDS / f"{args.arm}.jsonl"
    t0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if args.listed:
                content = LISTED_PROMPT.format(queues="\n".join(allowed), text=r["text"])
            else:
                content = TRAINED_PROMPT.format(text=r["text"])
            ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                          add_generation_prompt=True, enable_thinking=False)
            s = time.time()
            out, logp = greedy(model, eos, ids)
            raw = tok.decode(out)
            f.write(json.dumps({
                "text": r["text"], "label": r["label"], "raw": raw,
                "pred": normalise(raw, allowed_set), "confidence": round(math.exp(logp), 6),
                "prompt_tokens": len(ids), "seconds": round(time.time() - s, 4),
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 250 == 0:
                print(f"{args.arm}: {i + 1}/{len(rows)}", flush=True)
    wall = time.time() - t0
    preds = read_jsonl(path)
    acc = sum(p["pred"] == p["label"] for p in preds) / len(preds)
    print(f"{args.arm}: accuracy {acc:.3f} on {len(preds)}, {wall:.0f}s, {len(preds) / wall:.1f} msg/s")
    runs = PREDS.parent / "local_runs.jsonl"
    with open(runs, "a") as f:
        f.write(json.dumps({"arm": args.arm, "test": args.test, "day": args.day,
                            "adapter": args.adapter, "listed": args.listed, "n": len(preds),
                            "wall_seconds": round(wall, 1), "accuracy": round(acc, 4)}) + "\n")


if __name__ == "__main__":
    main()
