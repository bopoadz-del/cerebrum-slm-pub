# Router run v5 (label set v9, 22 labels)

- Started: 2026-10-07T18:09:03+0400 (box time, +04). Finished 18:21:47 +04.
- Data: main at 0e81e95 (0e81e954a37c56813e2bde038d1fbc70818d5534, PR #43, pr7). train 1602 / val 880, exactly as merged, rebuilt with the v9 prompt (1602/880 system prompts equal prompt_for v9, user text unchanged). Change request: train 24, val 4. Every v9 label has val n>0.
- First attempt on a35e160 stopped at preflight. One train row was about 24k tokens (base64 archive bytes, label `other`): train_max 26581, 1 row over 6656. Nothing was trained and nothing was spent (`preflight_a35e160_wall.json`). pr7 fixed it with a general content rule: text that is mostly encoded bytes has no content (`router.content.is_mostly_encoded`, runs of 64+ characters, ratio > 0.5). The split was re-run with the same seed and caps. No limit was raised and no row was dropped by id.
- Preflight on 0e81e95: train_max 6387, 0 rows over 6656, val prompt max 6470 (`preflight.json`).
- Recipe (same as v4): Qwen/Qwen3.5-4B LoRA r16, alpha 16, 2 epochs, lr 4.73e-4, batch 16, seed 0, 202 steps, max_sequence_length 6656. v3 used 6144; v4 raised it to 6656 because the v9 prompt is longer (longest row 6387 tokens). No row was dropped or cut.
- Tinker run: 4ec9bca9-f838-5d1a-ad89-7157820a3f09:train:0
- Checkpoints:
  - epoch 1: tinker://4ec9bca9-f838-5d1a-ad89-7157820a3f09:train:0/sampler_weights/router-v5-qwen3.5-4b-lora-r16-epoch1
  - epoch 2: tinker://4ec9bca9-f838-5d1a-ad89-7157820a3f09:train:0/sampler_weights/router-v5-qwen3.5-4b-lora-r16-epoch2
- **Chosen epoch: 1.** It has the higher val macro recall (0.6579 vs 0.5317), and the choice was made before any sealed score. Epoch 2 is 0.005 higher on accuracy only because drawing (n=658) and report improve. It loses payment application, payment certificate, specification, inspection request and other (`epoch_choice.json`). The top-level fields of `metrics.json` show the final epoch (2) by default.
- Sealed sets: not read, not scored. Sealed scoring waits for the owner's ticks on the 14 change-request candidates and a new order.

## Val (n=880)

| | epoch 1 | epoch 2 |
| --- | ---: | ---: |
| macro recall (all 22 labels, n>0) | 0.6579 | 0.5317 |
| accuracy | 0.8602 | 0.8648 |
| val gate (beats majority, majority fails) | True | True |
| unsure/off-list | 0 | 0 |

Majority baseline: drawing, accuracy 0.7477, macro 0.0455.

| label | val n | epoch 1 recall | epoch 2 recall |
| --- | ---: | ---: | ---: |
| RFI | 7 | 0.143 | 0.143 |
| submittal | 4 | 1.000 | 1.000 |
| transmittal | 1 | 1.000 | 1.000 |
| VO/CO | 1 | 0.000 | 0.000 |
| change request | 4 | 1.000 | 1.000 |
| claim | 1 | 0.000 | 0.000 |
| payment application | 1 | 1.000 | 0.000 |
| payment certificate | 2 | 1.000 | 0.000 |
| drawing | 658 | 0.986 | 0.998 |
| specification | 7 | 0.286 | 0.000 |
| schedule | 8 | 1.000 | 0.875 |
| report | 43 | 0.256 | 0.860 |
| correspondence | 10 | 0.100 | 0.100 |
| BOQ | 8 | 0.750 | 0.875 |
| contract | 11 | 0.545 | 0.545 |
| method statement | 5 | 1.000 | 1.000 |
| meeting minutes | 10 | 1.000 | 1.000 |
| NCR | 2 | 1.000 | 1.000 |
| inspection request | 7 | 1.000 | 0.143 |
| site instruction | 8 | 0.000 | 0.000 |
| register/log | 6 | 1.000 | 1.000 |
| other | 76 | 0.408 | 0.158 |

Never predicted on val: epoch 1 VO/CO, claim, site instruction; epoch 2 VO/CO, claim, payment application, payment certificate, specification, site instruction. Several classes have very small val n (VO/CO 1, claim 1, transmittal 1, payment application 1), so their recall moves in steps of 1.0. Do not compare this macro directly with v4's 0.6389: v4 averaged over 18 labels on 848 val rows, v5 over 22 labels on 880.
