"""Write the queue context the follow-up Claude arms see beside each name.

    python write_context.py

Both kinds come from training messages only, never the test set:

- descriptions: Claude Opus 5.5 reads the 70 queue names and the 2,240
  messages in train_old_k32.jsonl, grouped by queue, and writes one line
  per queue. One call, asked again up to twice if the reply is unusable.
- examples: three messages per queue drawn with a fixed seed from
  train_old_k8.jsonl, the 560 the 8-label adapter learned from.

Writes data/context/descriptions.json, descriptions_raw.json and
examples.json. Saved descriptions are kept on a rerun, never asked for again.
"""

import json
import random
from collections import defaultdict

from route import ROOT, SPLITS, queues, read_jsonl
from score_claude import call

CONTEXT = ROOT / "data" / "context"
WRITER = "claude-opus-5-5"
MAX_WORDS = 25
MAX_RETRIES = 2
EXAMPLE_SEED = 3
EXAMPLES_PER_QUEUE = 3
BANNED = (chr(0x2014), chr(0x2015))  # em dash, horizontal bar

SYSTEM = "You write routing notes for a bank's contact centre. Answer with JSON only."
USER = (
    "A contact centre routes customer messages into the {n} queues below. Under each queue name "
    "are past messages that staff routed to it.\n\n"
    "Write one line for each queue, at most 20 words, saying what kind of message belongs in it, "
    "so that someone who has never seen these messages could route new ones correctly. Where two "
    "queues look alike, say what separates them. Use plain words and no dashes.\n\n"
    "Reply with one JSON object that maps each of the {n} queue names, spelled exactly as below, "
    "to its line, and nothing else.\n\n{queues}"
)


def by_queue(split):
    out = defaultdict(list)
    for r in read_jsonl(SPLITS / split):
        out[r["label"]].append(r["text"])
    return out


def writer_prompt(allowed):
    msgs = by_queue("train_old_k32.jsonl")
    blocks = ["## " + q + "\n" + "\n".join(f"- {json.dumps(t, ensure_ascii=False)}" for t in msgs[q])
              for q in allowed]
    return USER.format(n=len(allowed), queues="\n\n".join(blocks))


def objects(text):
    """Every JSON object in a reply, in the order they appear."""
    dec, found = json.JSONDecoder(), []
    i = text.find("{")
    while i != -1:
        try:
            value, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        i = text.find("{", end)
    return found


def usable(reply, allowed):
    """The last object in a reply with exactly the queue names as keys, a line of at most
    MAX_WORDS words for each, and no em dash; otherwise None. The writer is asked for 20
    words and held to 25, because models count words loosely."""
    for obj in reversed(objects(reply or "")):
        if set(obj) != set(allowed):
            continue
        lines = [obj[q] for q in allowed]
        if all(isinstance(v, str) and 0 < len(v.split()) <= MAX_WORDS and not any(b in v for b in BANNED)
               for v in lines):
            return {q: obj[q].strip() for q in allowed}
    return None


def draw_examples(allowed, seed=EXAMPLE_SEED, k=EXAMPLES_PER_QUEUE):
    msgs = by_queue("train_old_k8.jsonl")
    rng = random.Random(seed)
    return {q: rng.sample(msgs[q], k) for q in allowed}


def write_descriptions(allowed):
    target = CONTEXT / "descriptions.json"
    if target.exists():
        print(f"kept {target.relative_to(ROOT)}")
        return
    prompt, attempts = writer_prompt(allowed), []
    for k in range(MAX_RETRIES + 1):
        response = call(WRITER, None, prompt, system=SYSTEM, timeout=1200)
        attempts.append({"attempt": k, "model": WRITER, "response": response})
        # ASCII-escaped, so the raw reply is kept exactly without a stray em dash in the repo.
        (CONTEXT / "descriptions_raw.json").write_text(json.dumps(attempts, indent=1) + "\n")
        lines = usable(response.get("result", ""), allowed)
        print(f"descriptions attempt {k}: {'usable' if lines else 'unusable'}", flush=True)
        if lines:
            target.write_text(json.dumps({
                "writer": WRITER, "source": "data/splits/train_old_k32.jsonl", "attempts": k + 1,
                "descriptions": lines}, indent=2, ensure_ascii=False) + "\n")
            return
    raise SystemExit("no usable descriptions after three attempts; see descriptions_raw.json")


def main():
    allowed = queues(1)
    CONTEXT.mkdir(parents=True, exist_ok=True)
    (CONTEXT / "examples.json").write_text(json.dumps({
        "source": "data/splits/train_old_k8.jsonl", "seed": EXAMPLE_SEED,
        "examples": draw_examples(allowed)}, indent=2, ensure_ascii=True) + "\n")
    write_descriptions(allowed)


if __name__ == "__main__":
    main()
