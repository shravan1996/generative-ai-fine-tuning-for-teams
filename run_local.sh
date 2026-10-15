#!/bin/zsh
# Every local step of the trial, in order. Each step is skipped when its
# output already exists, so the script can be restarted after a stop.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

score() {  # arm test day [extra args]
  [[ -f results/preds/$1.jsonl && $(wc -l < results/preds/$1.jsonl) -gt 0 ]] && return
  $PY score_local.py --arm $1 --test $2 --day $3 "${@:4}"
}
train() {  # train-split out [dev]
  [[ -f $2/run.json ]] && return
  $PY train.py --train $1 --out $2 --dev ${3:-dev_old.jsonl}
}

score qwen_listed_day1 test_old.jsonl 1 --listed
train train_old_k8.jsonl adapters/day1_k8
score qwen_k8_day1 test_old.jsonl 1 --adapter adapters/day1_k8
train train_old_k32.jsonl adapters/day1_k32
score qwen_k32_day1 test_old.jsonl 1 --adapter adapters/day1_k32
train train_old_all.jsonl adapters/day1_all
score qwen_all_day1 test_old.jsonl 1 --adapter adapters/day1_all
score qwen_all_day1_on_day90 test_all.jsonl 90 --adapter adapters/day1_all
train train_all_plus_new_k8.jsonl adapters/day90_retrain dev_all.jsonl
score qwen_retrain_day90 test_all.jsonl 90 --adapter adapters/day90_retrain
score qwen_listed_day90 test_all.jsonl 90 --listed
echo "local trial finished $(date)"
