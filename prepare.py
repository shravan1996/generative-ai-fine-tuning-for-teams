"""Build every split the trial uses from the BANKING77 CSVs.

Nothing here looks at a model. The seven "new" queues and every
training sample are drawn with fixed seeds, so a reader rebuilds the
exact files.

    python prepare.py
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "splits"

NEW_QUEUE_SEED = 77
NEW_QUEUE_COUNT = 7
DEV_PER_QUEUE = 3
SAMPLE_SEED = 2026
LABEL_BUDGETS = [8, 32]  # labels per queue; "all" is everything left after dev
RETRAIN_PER_NEW_QUEUE = 8


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["text"].strip(), row["category"].strip()) for row in csv.DictReader(f)]


def by_label(rows):
    groups = defaultdict(list)
    for text, label in rows:
        groups[label].append(text)
    return groups


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, label in rows:
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


def main():
    train = read_csv(RAW / "train.csv")
    test = read_csv(RAW / "test.csv")
    queues = sorted({label for _, label in train})
    assert len(queues) == 77, len(queues)

    new_queues = sorted(random.Random(NEW_QUEUE_SEED).sample(queues, NEW_QUEUE_COUNT))
    old_queues = [q for q in queues if q not in new_queues]

    rng = random.Random(SAMPLE_SEED)
    groups = by_label(train)
    dev, pool = [], {}
    for q in queues:
        texts = sorted(groups[q])
        rng.shuffle(texts)
        dev += [(t, q) for t in texts[:DEV_PER_QUEUE]]
        pool[q] = texts[DEV_PER_QUEUE:]

    # Day one: the team trains on the seventy queues it has today.
    # Budgets are nested, so the 8-label set is inside the 32-label set.
    for k in LABEL_BUDGETS:
        write_jsonl(OUT / f"train_old_k{k}.jsonl", [(t, q) for q in old_queues for t in pool[q][:k]])
    write_jsonl(OUT / "train_old_all.jsonl", [(t, q) for q in old_queues for t in pool[q]])
    write_jsonl(OUT / "dev_old.jsonl", [r for r in dev if r[1] in old_queues])

    # Day ninety: seven queues arrive, with a handful of labels each.
    new_labels = [(t, q) for q in new_queues for t in pool[q][:RETRAIN_PER_NEW_QUEUE]]
    write_jsonl(OUT / "train_new_k8.jsonl", new_labels)
    write_jsonl(OUT / "train_all_plus_new_k8.jsonl",
                [(t, q) for q in old_queues for t in pool[q]] + new_labels)
    write_jsonl(OUT / "dev_all.jsonl", dev)

    write_jsonl(OUT / "test_old.jsonl", [r for r in test if r[1] in old_queues])
    write_jsonl(OUT / "test_new.jsonl", [r for r in test if r[1] in new_queues])
    write_jsonl(OUT / "test_all.jsonl", test)

    meta = {
        "source": "BANKING77, PolyAI (CC BY 4.0)",
        "train_rows": len(train),
        "test_rows": len(test),
        "new_queues": new_queues,
        "old_queues": old_queues,
        "label_budgets": LABEL_BUDGETS + ["all"],
        "dev_per_queue": DEV_PER_QUEUE,
        "retrain_per_new_queue": RETRAIN_PER_NEW_QUEUE,
        "seeds": {"new_queues": NEW_QUEUE_SEED, "samples": SAMPLE_SEED},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("train_rows", "test_rows", "new_queues")}, indent=2))
    for p in sorted(OUT.glob("*.jsonl")):
        print(p.name, sum(1 for _ in open(p)))


if __name__ == "__main__":
    main()
