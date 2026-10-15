# generative-ai-fine-tuning-for-teams

A small, repeatable test of one question a team in a big company keeps meeting: should it
fine-tune a small open model for a narrow routing job, or keep sending everything to a
frontier model with the list of categories in the prompt? And what happens when the
categories change?

It is the trial behind the Generative AI article
[Seven queues it had never seen](https://shravanpendem.com/blog/generative-ai/operations-llmops/fine-tuning-small-models-for-teams)
on shravanpendem.com.

## The job

A contact-centre team at a bank routes customer messages into queues. The data is
[BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (PolyAI, CC BY 4.0): 10,003
training and 3,080 test messages over 77 intents. On day one the team has 70 queues. On day
ninety the business adds seven, drawn with a fixed seed before any model ran.

| Arm | What it is |
| --- | --- |
| Claude Haiku 5.5, Claude Sonnet 5.5 | prompted with the queue names, 20 messages per call |
| Qwen3.5-0.8B, prompted | the same queue list, no training |
| Qwen3.5-0.8B + LoRA | trained on 8, 32, or every labelled message per queue |
| Qwen3.5-0.8B retrained | day ninety: every old label plus 8 per new queue |

The plan, thresholds and arms were committed in [PREREGISTRATION.md](PREREGISTRATION.md)
before any arm was scored.

## Results

| Arm | Day one, 70 queues | Day ninety, old queues | Day ninety, new queues |
| --- | --- | --- | --- |
| Qwen3.5-0.8B, queue list, untrained | 43.5% |  |  |
| Claude Haiku 5.5, queue list, low effort | 76.0% | 71.8% | 75.4% |
| Claude Sonnet 5.5, queue list | 81.5% | 79.5% | 86.8% |
| Qwen3.5-0.8B + 8 labels per queue | 80.4% |  |  |
| Qwen3.5-0.8B + 32 labels per queue | 86.9% |  |  |
| Qwen3.5-0.8B + every label | 93.1% | 93.1% | 2.1% |
| Qwen3.5-0.8B retrained, + 8 per new queue |  | 93.2% | 60.7% |
| Qwen3.5-0.8B, queue list, untrained (77 names) |  | 41.7% | 47.1% |
| Exploratory: every-label Qwen, Sonnet used only for new queues |  | 90.9% | 86.8% |

Accuracy is the share of test messages routed to the right queue: 2,800 messages on day one,
and on day ninety 2,800 for the old queues and 280 for the seven new ones. Each figure has a
95% interval of about 1.5 points on day one and about 5 points for the new queues.

What the pre-registered questions came to:

1. **A few days of labels beat Claude given the queue names.** 32 labels per queue scored
   86.9% against 76.0% for Haiku and 81.5% for Sonnet. 8 per queue (80.4%) was level with
   Sonnet (paired test p = 0.16).
2. **The trained model's confidence barely flags new queues.** AUROC 0.81 against a 0.80 bar.
   A threshold that sends 10% of ordinary messages to a person caught 149 of 280 new-queue
   messages, and 69 new-queue answers came with 90% confidence or more, all wrong.
3. **Eight labels per new queue were not enough.** The retrained model kept the old queues
   (93.2%) but got 60.7% of the new ones, under the 70% bar and under both Claude models.

The exploratory row was worked out after the results, from saved predictions: Claude reads
every message with the full list, and its answer is used only when it names a queue the
small model was never trained on. One deviation from the plan, how Claude's replies are
parsed, is recorded at the end of the pre-registration.

Training on an M4 Pro Mac with 24 GB: 17 minutes for 8 labels per queue, 22 for 32, about two
hours for every label, peak memory 5.5 to 6.5 GB. At list price, routing 10,000 messages cost
about $0.38 on Haiku and $4.87 on Sonnet in these runs.

Every number above is in [results/summary.json](results/summary.json), computed by
`analyze.py` from the saved predictions in `results/preds/`. Raw Claude replies, with the
token counts and list-price cost each call reported, are in `results/claude_raw/`.

## Check it without a key

Needs Python 3.10 or later and nothing else.

```bash
git clone https://github.com/shravan1996/generative-ai-fine-tuning-for-teams
cd generative-ai-fine-tuning-for-teams
python3 -m unittest discover -s tests -t .
python3 analyze.py
```

The tests rebuild every split from the CSVs and compare them byte for byte, check that the
seven new queues never appear in day-one training data, check the label matching and the
statistics, and confirm that the committed summary follows from the committed predictions.

## Run it again

Training and local scoring need an Apple-silicon Mac (MLX). The Claude arms need the Claude
Code CLI signed in, or edit `score_claude.py` to call the API directly.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare.py
./run_local.sh        # trains four adapters and scores six arms; hours on a laptop
./run_claude.sh       # four Claude arms and a batching check
.venv/bin/python analyze.py
```

Three things that cost me time and might cost you some:

- **Training mode matters for Qwen3.5.** Its linear-attention layers use a fast Metal kernel
  that has no backward pass. mlx-lm switches to a slower reference path in training mode,
  which `train()` sets for you; a hand-written loop that forgets `model.train()` fails with
  `[Primitive::vjp] Not implemented for CustomKernel`.
- **The reference path is memory-hungry.** Without gradient checkpointing, 16 messages of 48
  tokens needed 35 GB. With checkpointing and micro-batches of 4, training peaks near 5 GB.
- **One BANKING77 queue is spelled with a capital letter**, `Refund_not_showing_up`. A
  scorer that lowercases answers marks every correct reply for that queue wrong.

## Files

| Path | What it holds |
| --- | --- |
| `prepare.py` | builds every split in `data/splits/` from `data/raw/` |
| `route.py` | the two prompts and the queue-name matching |
| `train.py` | LoRA training with fixed settings |
| `score_local.py`, `score_claude.py` | route the test messages and save every answer |
| `analyze.py` | accuracy, intervals, McNemar tests, the new-queue alarm, cost |
| `make_figures.py` | the article's figures, from `results/summary.json` |
| `run_local.sh`, `run_claude.sh`, `guard.sh` | the full run, and a disk and swap guard |

## Licence

Code: MIT. BANKING77 is © PolyAI, released under CC BY 4.0; the copy in `data/raw/` is
unchanged and carries its licence. Qwen3.5-0.8B is Apache 2.0 and is downloaded at run time.
