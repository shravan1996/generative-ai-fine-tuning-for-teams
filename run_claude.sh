#!/bin/zsh
# Every Claude arm of the trial. Saved replies are reused on a rerun.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY score_claude.py --arm haiku_day1 --model claude-haiku-5-5 --effort low --test test_old.jsonl --day 1
$PY score_claude.py --arm sonnet_day1 --model claude-sonnet-5-5 --test test_old.jsonl --day 1
$PY score_claude.py --arm haiku_day90 --model claude-haiku-5-5 --effort low --test test_all.jsonl --day 90
$PY score_claude.py --arm sonnet_day90 --model claude-sonnet-5-5 --test test_all.jsonl --day 90
# Batching check: the first 200 day-one messages in the same shuffled order, one per call.
$PY score_claude.py --arm haiku_day1_single --model claude-haiku-5-5 --effort low --test test_old.jsonl --day 1 --batch-size 1 --limit-batches 200
echo "claude trial finished $(date)"
