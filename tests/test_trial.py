"""Checks a reader can run with the standard library only: no model, no key.

    python3 -m unittest discover -s tests -t .
"""

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
from score_claude import batches, parse  # noqa: E402

SPLITS = ROOT / "data" / "splits"


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
        k8 = {(r["text"], r["label"]) for r in read_jsonl(SPLITS / "train_old_k8.jsonl")}
        k32 = {(r["text"], r["label"]) for r in read_jsonl(SPLITS / "train_old_k32.jsonl")}
        full = {(r["text"], r["label"]) for r in read_jsonl(SPLITS / "train_old_all.jsonl")}
        self.assertEqual((len(k8), len(k32)), (560, 2240))
        self.assertTrue(k8 <= k32 <= full)

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

    def test_analysis_reruns_offline(self):
        if not (ROOT / "results" / "summary.json").exists():
            self.skipTest("no recorded results yet")
        with tempfile.TemporaryDirectory():
            out = subprocess.run([sys.executable, "analyze.py"], cwd=ROOT, check=True,
                                 capture_output=True, text=True).stdout
        self.assertIn("day1", json.loads(out))


if __name__ == "__main__":
    unittest.main()
