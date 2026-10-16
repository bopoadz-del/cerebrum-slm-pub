# Run 4 (r16, cap-3 paraphrases, originals x3 loss weight): protocol-v2 scores and STOP GATE

Aggregate counts only. No eval item text.

## Run

- Data/config from main dcac5c7 (#20): paraphrase children capped at **3/parent** (deterministic sha256 order); ORIGINAL_LOSS_WEIGHT = 3.0 on assistant tokens for originals; paraphrases/hard-negatives at 1.0; **no** Factory Domain Intake pairs.
- Train pairs: **957** = 209 confirmed originals + 24 temporary-works + 699 paraphrases + 25 hard negatives.
- Config configs/lora/qwen3-8b-posture-r16.json: Qwen3-8B, rank 16, LR 0.000473, 2 epochs, batch 16, **120 steps**.
- Tinker run d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0.
- Epoch samplers:
  - tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch1
  - tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch2
- Final: tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/weights/posture-qwen3-8b-lora-r16 and tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16.
- Scored under protocol v2 (routing with attached formula registry; dual sample). Gate uses the protocol-v2 prompt-only bar in eval/baselines/scores.md (held route 61, blind route 17).

## Pair counts by source

| Source | Count |
|---|---|
| original_confirmed_train | 209 |
| temporary_works_train | 24 |
| paraphrase_children (3/parent) | 699 |
| hard_negatives | 25 |
| **train total** | **957** |

## Scores

Held-out n=69, blind n=60. RQ = refusal/qualification passes, Route = formula-routing passes, Leak = fact-leak count.

| Model | Held-out RQ | Held-out Route | Held-out Leak | Blind RQ | Blind Route | Blind Leak |
|---|---|---|---|---|---|---|
| Prompt-only (protocol v2) | 51 | 61 | 0 | 3 | 17 | 0 |
| Epoch 1 tuned (reported) | 0 | 0 | 20 | 6 | 0 | 29 |
| Epoch 1 tuned + prompt (gated) | 62 | 60 | 0 | 3 | 16 | 0 |
| Epoch 2 tuned (reported) | 0 | 0 | 23 | 3 | 0 | 28 |
| Epoch 2 tuned + prompt (gated) | 62 | 68 | 0 | 7 | 22 | 0 |

## STOP GATE (eval/STOP_GATE.md)

| Checkpoint | leak held | leak blind | RQ held | RQ blind | Route held | Route blind | Result |
|---|---|---|---|---|---|---|---|
| Epoch 1 tuned + prompt | ok | ok | ok (62>51) | fail (3=3) | fail (60!>61) | fail (16!>17) | FAIL |
| Epoch 2 tuned + prompt | ok | ok | ok (62>51) | ok (7>3) | ok (68>61) | ok (22>17) | **PASS** |

Decision: **PASS** on epoch 2 tuned + system prompt. Serve:

tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch2
