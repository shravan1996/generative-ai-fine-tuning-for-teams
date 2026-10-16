"""Print the README's results table from results/summary.json."""

import json
from pathlib import Path

s = json.loads((Path(__file__).parent / "results" / "summary.json").read_text())
pct = lambda x: f"{100 * x:.1f}%"

rows = [
    ("Qwen3.5-0.8B, queue list, untrained", "qwen_listed_day1", None),
    ("Claude Haiku 5.5, queue list, low effort", "haiku_day1", "haiku_day90"),
    ("Claude Sonnet 5.5, queue list", "sonnet_day1", "sonnet_day90"),
    ("Follow-up: Claude Haiku 5.5, a line on each queue", "haiku_day1_described", None),
    ("Follow-up: Claude Sonnet 5.5, a line on each queue", "sonnet_day1_described", None),
    ("Follow-up: Claude Haiku 5.5, three examples per queue", "haiku_day1_examples", None),
    ("Follow-up: Claude Sonnet 5.5, three examples per queue", "sonnet_day1_examples", None),
    ("Qwen3.5-0.8B + 8 labels per queue", "qwen_k8_day1", None),
    ("Qwen3.5-0.8B + 32 labels per queue", "qwen_k32_day1", None),
    ("Qwen3.5-0.8B + every label", "qwen_all_day1", "qwen_all_day1_on_day90"),
    ("Qwen3.5-0.8B retrained, + 8 per new queue", None, "qwen_retrain_day90"),
    ("Qwen3.5-0.8B, queue list, untrained (77 names)", None, "qwen_listed_day90"),
]
print("| Arm | Day one, 70 queues | Day ninety, old queues | Day ninety, new queues |")
print("| --- | --- | --- | --- |")
for name, d1, d90 in rows:
    a = pct(s["day1"][d1]["accuracy"]) if d1 and d1 in s["day1"] else ""
    if d90 and d90 in s["day90"]:
        b, c = pct(s["day90"][d90]["old"]["accuracy"]), pct(s["day90"][d90]["new"]["accuracy"])
    else:
        b = c = ""
    if a or b:
        print(f"| {name} | {a} | {b} | {c} |")
w = s.get("new_queue_watch_exploratory", {}).get("sonnet_day90")
if w:
    print(f"| Exploratory: every-label Qwen, Sonnet used only for new queues |  | "
          f"{pct(w['accuracy_old'])} | {pct(w['accuracy_new'])} |")
