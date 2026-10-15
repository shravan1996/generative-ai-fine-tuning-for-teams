"""Prompts and queue-name handling shared by training and scoring.

Two prompts exist. The fine-tuned model sees only the message, because
it has learned the queues. A prompted model, small or frontier, sees
the message and the list of queue names, which is all a team has on
the day it starts.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
SPLITS = ROOT / "data" / "splits"

TRAINED_PROMPT = (
    "Route this bank customer message to one queue. "
    "Reply with the queue name only.\n\nMessage: {text}"
)

LISTED_PROMPT = (
    "Route this bank customer message to exactly one of these queues. "
    "Reply with the queue name only.\n\nQueues:\n{queues}\n\nMessage: {text}"
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def meta():
    return json.loads((SPLITS / "meta.json").read_text())


def queues(day):
    m = meta()
    if day == 1:
        return list(m["old_queues"])
    if day == 90:
        return sorted(m["old_queues"] + m["new_queues"])
    raise ValueError(day)


def normalise(answer, allowed):
    """Map a model's reply onto a queue name, or None.

    Only cosmetic repairs: case, surrounding quotes and punctuation,
    spaces for underscores. A reply that names no queue counts as wrong.
    BANKING77 spells one queue "Refund_not_showing_up", so matching
    ignores case and returns the dataset's own spelling.
    """
    if not answer:
        return None
    canonical = {q.lower(): q for q in allowed}
    a = answer.strip().splitlines()[0] if answer.strip() else ""
    a = a.strip().strip("`'\".,:;[]").strip().lower().replace(" ", "_")
    return canonical.get(a)
