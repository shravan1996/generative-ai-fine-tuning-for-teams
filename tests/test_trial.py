"""Checks a reader can run with the standard library only: no model, no key.

    python3 -m unittest discover -s tests -t .
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import analyze  # noqa: E402
from route import normalise, queues, read_jsonl  # noqa: E402
import write_context  # noqa: E402
from score_claude import batches, build_prompt, parse  # noqa: E402

SPLITS = ROOT / "data" / "splits"
CONTEXT = ROOT / "data" / "context"


class Splits(unittest.TestCase):
    def test_rebuild_is_identical(self):
        """prepare.py rebuilds every split byte for byte from the CSVs."""
        before = {p.name: p.read_bytes() for p in SPLITS.iterdir()}
        subprocess.run([sys.executable, "prepare.py"], cwd=ROOT, check=True, capture_output=True)
        after = {p.name: p.read_bytes() for p in SPLITS.iterdir()}
        self.assertEqual(before, after)

    def test_new_queues_never_trained_on_day_one(self):
        new = set(json.loads((SPLITS / "meta.json").read_text())["new_queues"])
        self.assertEqual(len(new), 7)
        for name in ("train_old_k8", "train_old_k32", "train_old_all", "dev_old", "test_old"):
            labels = {r["label"] for r in read_jsonl(SPLITS / f"{name}.jsonl")}
            self.assertFalse(labels & new, name)
            self.assertEqual(len(labels), 70, name)

    def test_budgets_are_nested(self):
        rows = {n: read_jsonl(SPLITS / f"train_old_{n}.jsonl") for n in ("k8", "k32", "all")}
        self.assertEqual((len(rows["k8"]), len(rows["k32"])), (560, 2240))
        # BANKING77 repeats a few messages word for word, so compare as sets.
        pairs = {n: {(r["text"], r["label"]) for r in v} for n, v in rows.items()}
        self.assertTrue(pairs["k8"] <= pairs["k32"] <= pairs["all"])

    def test_retrain_adds_eight_per_new_queue(self):
        rows = read_jsonl(SPLITS / "train_new_k8.jsonl")
        counts = {}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        self.assertEqual(set(counts.values()), {8})
        self.assertEqual(len(counts), 7)

    def test_test_set_shape(self):
        self.assertEqual(len(read_jsonl(SPLITS / "test_old.jsonl")), 2800)
        self.assertEqual(len(read_jsonl(SPLITS / "test_new.jsonl")), 280)


class Matching(unittest.TestCase):
    def test_case_and_dataset_spelling(self):
        allowed = set(queues(1))
        self.assertEqual(normalise("refund_not_showing_up", allowed), "Refund_not_showing_up")
        self.assertEqual(normalise(' "Card Arrival". ', allowed), "card_arrival")
        self.assertIsNone(normalise("card_linking", allowed))  # a day-ninety queue
        self.assertIsNone(normalise("", allowed))

    def test_claude_reply_parsing(self):
        self.assertEqual(parse('```json\n["a", "b"]\n```', 2), ["a", "b"])
        self.assertEqual(parse('["a"]', 2), ["a", None])
        self.assertEqual(parse("no json here", 2), [None, None])

    def test_batches_cover_every_message_once(self):
        rows = list(range(2800))
        plan = batches(rows, 20)
        flat = [i for b in plan for i in b]
        self.assertEqual(sorted(flat), rows)
        self.assertEqual(len(plan), 140)


class QueueContext(unittest.TestCase):
    """The follow-up arms change the queue list and nothing else."""

    def test_names_prompt_is_what_the_first_arms_saw(self):
        rows = read_jsonl(SPLITS / "test_old.jsonl")
        texts = [rows[j]["text"] for j in batches(rows, 20)[0]]
        prompt = build_prompt("names", queues(1), texts)
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(),
                         "cccae23bda04b1ed2fb5fe31863563c1e416bf8671c0d683ca402293e3307d53")

    def test_examples_follow_the_seed_from_eight_label_data(self):
        path = CONTEXT / "examples.json"
        if not path.exists():
            self.skipTest("no examples written yet")
        saved = json.loads(path.read_text())["examples"]
        self.assertEqual(saved, write_context.draw_examples(queues(1)))
        k8 = {(r["label"], r["text"]) for r in read_jsonl(SPLITS / "train_old_k8.jsonl")}
        for q, texts in saved.items():
            self.assertEqual(len(texts), 3, q)
            self.assertTrue(all((q, t) in k8 for t in texts), q)

    def test_descriptions_cover_every_queue_in_one_short_line(self):
        path = CONTEXT / "descriptions.json"
        if not path.exists():
            self.skipTest("no descriptions written yet")
        lines = json.loads(path.read_text())["descriptions"]
        self.assertEqual(list(lines), queues(1))
        test_texts = {r["text"].strip().lower() for r in read_jsonl(SPLITS / "test_old.jsonl")}
        for q, line in lines.items():
            self.assertLessEqual(len(line.split()), write_context.MAX_WORDS, q)
            self.assertEqual("\n" in line, False, q)
            self.assertFalse(any(b in line for b in write_context.BANNED), q)
            # No test message is quoted, word for word, in a description.
            self.assertFalse([t for t in test_texts if len(t) >= 25 and t in line.lower()], q)
        prompt = build_prompt("descriptions", queues(1), ["hello"])
        self.assertTrue(all(f"{q}: {line}" in prompt for q, line in lines.items()))

    def test_description_reply_checks(self):
        allowed = ["a", "b"]
        good = '{"a": "one line", "b": "another line"}'
        self.assertEqual(write_context.usable("Sure.\n" + good, allowed), {"a": "one line", "b": "another line"})
        self.assertIsNone(write_context.usable('{"a": "one line"}', allowed))
        self.assertIsNone(write_context.usable('{"a": "x", "b": "' + "word " * 26 + '"}', allowed))
        self.assertIsNone(write_context.usable('{"a": "x", "b": "y ' + write_context.BANNED[0] + ' z"}', allowed))


class Statistics(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(analyze.auroc([3, 4], [1, 2]), 1.0)
        self.assertEqual(analyze.auroc([1, 2], [3, 4]), 0.0)
        self.assertEqual(analyze.auroc([1], [1]), 0.5)

    def test_wilson_contains_estimate(self):
        lo, hi = analyze.wilson(2464, 2800)
        self.assertLess(lo, 0.88)
        self.assertGreater(hi, 0.88)

    def test_mcnemar_symmetric_case(self):
        a = {0: {"pred": "x", "label": "x"}, 1: {"pred": "y", "label": "x"}}
        b = {0: {"pred": "y", "label": "x"}, 1: {"pred": "x", "label": "x"}}
        self.assertEqual(analyze.mcnemar(a, b)["p"], 1.0)


class RecordedResults(unittest.TestCase):
    """The committed summary must follow from the committed predictions."""

    def test_summary_matches_predictions(self):
        summary_path = ROOT / "results" / "summary.json"
        if not summary_path.exists():
            self.skipTest("no recorded results yet")
        summary = json.loads(summary_path.read_text())
        for arm, stats in summary["day1"].items():
            rows = read_jsonl(ROOT / "results" / "preds" / f"{arm}.jsonl")
            right = sum(r["pred"] == r["label"] for r in rows)
            self.assertEqual((stats["n"], stats["correct"]), (len(rows), right), arm)

    def test_flip_rule_follows_the_counts(self):
        summary = json.loads((ROOT / "results" / "summary.json").read_text())
        ctx = summary.get("queue_context")
        if not ctx:
            self.skipTest("no follow-up arms scored yet")
        self.assertEqual(ctx["reference_correct"], summary["day1"]["qwen_k32_day1"]["correct"])
        self.assertEqual(ctx["flip_if_correct_at_least"], ctx["reference_correct"] - 2 * ctx["n"] // 100)
        for arm, stats in ctx["arms"].items():
            self.assertEqual(stats["correct"], summary["day1"][arm]["correct"], arm)
            self.assertEqual(stats["flips_verdict"], stats["correct"] >= ctx["flip_if_correct_at_least"], arm)
        self.assertEqual(ctx["flip_condition_met"], any(s["flips_verdict"] for s in ctx["arms"].values()))

    def test_analysis_reruns_offline(self):
        if not (ROOT / "results" / "summary.json").exists():
            self.skipTest("no recorded results yet")
        with tempfile.TemporaryDirectory():
            out = subprocess.run([sys.executable, "analyze.py"], cwd=ROOT, check=True,
                                 capture_output=True, text=True).stdout
        self.assertIn("day1", json.loads(out))


if __name__ == "__main__":
    unittest.main()
