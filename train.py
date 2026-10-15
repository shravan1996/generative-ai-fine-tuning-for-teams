"""LoRA fine-tune of a small Qwen model on one training split.

    python train.py --train train_old_k32.jsonl --out adapters/day1_k32
    python train.py --train train_new_k8.jsonl --out adapters/day90_patch \
        --resume adapters/day1_all

Every run uses the same settings. The update count is a rule, not a
choice per run: two passes over the data, and never fewer than 200
updates, so the smallest budgets are not starved. Qwen3.5 mixes
attention with a linear-attention layer whose fast Metal kernel has no
backward pass, so training runs the slower reference path; micro-batches
of 4 with gradient checkpointing keep it near 5 GB on a 24 GB Mac.
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers

from route import SPLITS, TRAINED_PROMPT, read_jsonl

MODEL = "Qwen/Qwen3.5-0.8B"
SETTINGS = {
    "rank": 16,
    "scale": 2.0,
    "dropout": 0.05,
    "num_layers": -1,  # every transformer block
    "batch_size": 4,
    "grad_accumulation": 4,  # 16 messages per update
    "learning_rate": 2e-4,
    "warmup_updates": 20,
    "epochs": 2,
    "min_updates": 200,
    "max_seq_length": 256,  # longest training turn is 133 tokens
    "seed": 0,
}


def encode(tokenizer, rows):
    """Token ids plus the offset where the answer starts, so only the
    queue name and the end-of-turn token are trained on."""
    out = []
    for r in rows:
        prompt = [{"role": "user", "content": TRAINED_PROMPT.format(text=r["text"])}]
        full = prompt + [{"role": "assistant", "content": r["label"]}]
        p = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, enable_thinking=False)
        f = tokenizer.apply_chat_template(full, enable_thinking=False)
        assert f[: len(p)] == p, "prompt is not a prefix of the full turn"
        out.append((f, len(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default="dev_all.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None, help="adapter directory to continue from")
    args = ap.parse_args()

    s = SETTINGS
    mx.random.seed(s["seed"])
    model, tokenizer = load(MODEL)
    model.freeze()
    linear_to_lora_layers(model, s["num_layers"], {k: s[k] for k in ("rank", "scale", "dropout")})
    if args.resume:
        model.load_weights(str(Path(args.resume) / "adapters.safetensors"), strict=False)

    train_rows = read_jsonl(SPLITS / args.train)
    dev_rows = read_jsonl(SPLITS / args.dev)
    train_set = encode(tokenizer, train_rows)
    dev_set = encode(tokenizer, dev_rows)

    per_update = s["batch_size"] * s["grad_accumulation"]
    updates = max(s["min_updates"], math.ceil(s["epochs"] * len(train_set) / per_update))
    iters = updates * s["grad_accumulation"]
    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, s["learning_rate"], s["warmup_updates"]),
         optim.cosine_decay(s["learning_rate"], updates - s["warmup_updates"], 1e-6)],
        [s["warmup_updates"]],
    )
    optimizer = optim.Adam(learning_rate=schedule)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"{args.train}: {len(train_set)} rows, {updates} updates, {trainable / 1e6:.2f}M trainable parameters")

    t0 = time.time()
    train(
        model,
        optimizer,
        train_set,
        dev_set,
        args=TrainingArgs(
            batch_size=s["batch_size"],
            iters=iters,
            val_batches=-1,
            steps_per_report=100,
            steps_per_eval=max(100, iters // 4),
            steps_per_save=iters,
            max_seq_length=s["max_seq_length"],
            adapter_file=str(out / "adapters.safetensors"),
            grad_checkpoint=True,
            grad_accumulation_steps=s["grad_accumulation"],
        ),
    )
    seconds = time.time() - t0

    lora = {k: s[k] for k in ("rank", "scale", "dropout")}
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": s["num_layers"], "lora_parameters": lora,
        "model": MODEL,
    }, indent=2) + "\n")
    (out / "run.json").write_text(json.dumps({
        "model": MODEL, "train": args.train, "dev": args.dev, "resume": args.resume,
        "rows": len(train_set), "updates": updates, "settings": s,
        "trainable_parameters": int(trainable), "train_seconds": round(seconds, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }, indent=2) + "\n")
    print(f"done in {seconds / 60:.1f} min, peak {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
