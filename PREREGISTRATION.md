# Pre-registration

Written on 15 October 2026 at 22:35 IST, before any arm below was scored on the test set.

## Before this file

Two plumbing checks touched data: the prompted 0.8B model routed 20 development messages
(14 right), and Claude Haiku 5.5 routed the first two batches of the day-one test set,
40 messages (30 right). Both outputs were deleted. Every number in the article comes from
the full runs described here.

## The job

A contact-centre team at a large bank routes customer messages into queues. On day one it
has 70 queues. Ninety days later the business adds seven more. The data is BANKING77
(PolyAI, CC BY 4.0): 10,003 training and 3,080 test messages over 77 intents. The seven
"new" queues were drawn with `random.Random(77)` before any model ran:

`card_linking`, `contactless_not_working`, `country_support`, `disposable_card_limits`,
`exchange_charge`, `fiat_currency_support`, `getting_virtual_card`.

## Arms

Day one, 70 queues, scored on the 2,800 test messages whose intent is one of them:

| Arm | What it is |
| --- | --- |
| `haiku_day1` | Claude Haiku 5.5, effort low, prompted with the 70 queue names, 20 messages per call |
| `sonnet_day1` | Claude Sonnet 5.5, default effort, same prompt |
| `qwen_listed_day1` | Qwen3.5-0.8B, no training, same queue list in the prompt, one message per call |
| `qwen_k8_day1` | Qwen3.5-0.8B with a LoRA adapter trained on 8 labelled messages per queue (560) |
| `qwen_k32_day1` | the same with 32 per queue (2,240), roughly a week of one person's labelling |
| `qwen_all_day1` | the same with every remaining training label (9,024) |

Day ninety, 77 queues, scored on all 3,080 test messages, reported separately for the
2,800 old-queue and 280 new-queue messages:

| Arm | What it is |
| --- | --- |
| `haiku_day90`, `sonnet_day90` | the same Claude prompts with the 77 names |
| `qwen_all_day1_on_day90` | the day-one all-label adapter, unchanged |
| `qwen_retrain_day90` | a new adapter from the base model on every old label plus 8 per new queue (9,080) |
| `qwen_listed_day90` | the untrained 0.8B model with the 77 names |

Training settings are fixed for every adapter (see `train.py`): rank 16, scale 2.0, dropout
0.05, every layer, learning rate 2e-4 with 20 warmup updates and cosine decay, 16 messages
per update, two passes over the data and never fewer than 200 updates. No setting is tuned
on the test set. The development split (3 per queue) is only watched for loss.

## Questions and thresholds

**1. Does a week of labels buy Haiku's accuracy?** On the 2,800 day-one messages:

- *Counts as a tie:* `qwen_k32_day1` within 2 points of `haiku_day1` either way.
- *Reason to fine-tune:* `qwen_k32_day1` at least 2 points above `haiku_day1`.
- *Claim fails:* `qwen_all_day1` still more than 5 points below `haiku_day1`.

A paired McNemar test is reported beside each gap. With 2,800 messages, a 95% interval on
one accuracy is about 1.5 points wide on each side, which is why 2 points is the line.

**2. Does the trained model notice the new queues?** Using `qwen_all_day1_on_day90`, the
probability the model gave its own answer is the alarm. It is usable if it separates the 280
new-queue messages from the 2,800 old-queue ones with an AUROC of at least 0.80. I also report
how many new-queue messages a threshold catches when it sends 10% of old-queue traffic to a
person.

**3. Is a small relabel enough?** `qwen_retrain_day90` is good enough if it gets at least 70%
of new-queue messages right while losing no more than 2 points on old-queue messages against
`qwen_all_day1`. Compared with `haiku_day90` on both groups.

**Also recorded, not gated:** training minutes on an M4 Pro Mac with 24 GB, local routing
speed, and Claude's list-price cost per 10,000 messages from the token counts each call
reports.

## What would make me distrust the result

- BANKING77 has been public since 2020 and may sit in either model's training data.
- Claude routes 20 messages per call; the small model routes one. A check of 200 messages
  one at a time on Haiku will be reported beside the batched number.
- One seed, one split, one dataset. These are clean, short, English messages from one bank.

## Deviations

**15 October 2026, 22:50 IST: how Claude's replies are read.** The plan above did not say how
a reply is turned into 20 answers. The first parser took everything from the first `[` to the
last `]`. After `haiku_day1` and `sonnet_day1` finished, and before any fine-tuned arm had
been scored, I found two kinds of reply it misread:

- five Sonnet replies printed a list, then a line such as "Wait, I need to correct that", then
  a second list; the parser failed and counted all 20 messages as unanswered;
- three Haiku replies held a single queue name for 20 messages.

The parser now reads the last list in a reply that has exactly one name per message, and a
reply without one is asked again, up to twice. Every reply, first or retried, is kept under
`results/claude_raw/`, and its tokens count toward cost. The change can only raise Claude's
scores, which works against the fine-tuned arms. `usage.json` for each arm records how many
batches were retried and how many never produced a complete list.

## Follow-up: queue descriptions and examples

Written on 16 October 2026 at 06:47 IST, after every arm above was scored and the draft
article was written, and before any arm below touched the test set.

The draft's first condition for changing its verdict is a frontier model, given a one-line
description of each queue or a few examples per queue, coming within 2 points of the 32-label
model on the same 2,800 messages. The draft did not test it. These four arms do.

### What I already know

Every result above. In particular, both Claude models routed none of the 40
`get_physical_card` test messages correctly, and the draft quotes two of those messages,
which are about PINs. So I do not write the descriptions myself. A model writes them from
training messages only, with the prompt fixed in `write_context.py`, before any test message
is scored, and nobody edits them afterwards.

### Arms

| Arm | What it is |
| --- | --- |
| `haiku_day1_described` | Claude Haiku 5.5, effort low, each of the 70 queue names followed by a one-line description |
| `sonnet_day1_described` | Claude Sonnet 5.5, default effort, the same prompt |
| `haiku_day1_examples` | Claude Haiku 5.5, effort low, each name followed by three training messages from that queue |
| `sonnet_day1_examples` | Claude Sonnet 5.5, default effort, the same prompt |

Everything else matches `haiku_day1` and `sonnet_day1`: the same system prompt and
instruction, `test_old.jsonl`, the same shuffled batches of 20 (seed 7), the reply read as
the last list with one name per message, up to two retries, and every reply kept under
`results/claude_raw/`. Only the queue list and the line that introduces it change
(`build_prompt` in `score_claude.py`). The names-only prompt is byte for byte what the first
arms saw, and a test pins it.

**Descriptions.** Claude Opus 5.5, default effort, in one call through the Claude Code CLI,
reads the 70 queue names with all 2,240 messages from `data/splits/train_old_k32.jsonl` listed
under them, and is asked for one line per queue: at most 20 words on what belongs there, and
what separates queues that look alike. A reply is used if it holds a JSON object with exactly
the 70 names, every line at most 25 words, and no em dash; otherwise the same prompt is sent
again, up to twice. If no reply qualifies, the described arms are not run and I say so here.
The lines go to `data/context/descriptions.json` with the raw replies beside them, and are
committed before scoring. A model writes them because I have read test messages. A team would
write its own, and might do better or worse.

**Examples.** For each queue, in the order of `meta.json`, `random.Random(3).sample` picks
three of that queue's eight messages in `data/splits/train_old_k8.jsonl`: 210 messages, all
among the 560 the 8-label adapter was trained on. Saved to `data/context/examples.json`. They
are random, not chosen. Loukas et al. found expert-chosen examples scored about nine points
higher than random ones for GPT-4 on BANKING77, so this is the cheap version of the arm.

### Threshold

`qwen_k32_day1` routed 2,434 of the 2,800 messages correctly (86.93%). The draft's verdict
flips if any of the four arms routes at least 2,378 correctly (84.93%), that is, within 2
points of the 32-label model or above it. All four are reported whichever way it goes. Four
arms get four tries at the line, which slightly favours a flip; I note that rather than
correct for it, because the draft named both kinds of context.

Reported beside each arm, not gated:

- an exact paired McNemar test against `qwen_k32_day1`, and against the same model's
  names-only arm;
- right answers out of 40 for every queue, next to the names-only arms and the 32-label
  model, with `get_physical_card` called out;
- list-price cost per 10,000 messages from each arm's `usage.json`, and the cost of the call
  that wrote the descriptions.

The caution about BANKING77 sitting in training data applies to Opus as the writer too.

### Before scoring

One plumbing check: one batch of 20 messages from `dev_old.jsonl` per new arm, to see that
the prompts run and the replies parse. Its outputs are deleted and not reported. Nothing
touches `test_old.jsonl` until the descriptions and examples are committed.
