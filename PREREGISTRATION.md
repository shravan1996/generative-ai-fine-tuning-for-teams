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
