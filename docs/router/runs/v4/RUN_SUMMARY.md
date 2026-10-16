# Router run v4 (label set v9, 22 labels)

- Started: 2026-10-07T17:29:44+0400 (box time, +04). Eval-before-train chain: v3 epoch 1 was sealed-scored once and name-blind checked before this run (docs/router/runs/v3_sealed_v9/).
- Data: main at 868e5af (868e5aff8dbee6e7d0ef65b3f7f598a01ce248a4), train 1482 / val 848 (train.jsonl sha256 3bf6297e8d9c9149290cb4ce7bcc7cbd91ef5e416e6123d2c3a4b0810005772f, val.jsonl sha256 0a9501a2bf18725573aa5d94cf63ecfae33c84a87086c7f7ef19eaf636c34f00), exactly as merged, rebuilt with the v9 prompt (1482/848 system prompts equal prompt_for v9). pr5 (PR #42) was not merged when this run started, so main was used as is: the new change-request rows and the unmasked pr2 rows are not in this run. Change request: train 3, val 0.
- Recipe (same as v3): Qwen/Qwen3.5-4B LoRA r16, alpha 16, 2 epochs, lr 4.73e-4, batch 16, seed 0, 186 steps. max_sequence_length 6144 -> 6656. v3 used 6144 to fit its longest row (5979 tokens). The v9 prompt is longer, so the longest row is now 6387 tokens and 6 rows exceed 6144. The limit was raised to fit them; no row was dropped or cut.
- Tinker run: 3988e5b9-0c0b-53e8-a179-fefebbcb3e1c:train:0
- Checkpoints:
  - epoch 1: tinker://3988e5b9-0c0b-53e8-a179-fefebbcb3e1c:train:0/sampler_weights/router-v4-qwen3.5-4b-lora-r16-epoch1
  - epoch 2: tinker://3988e5b9-0c0b-53e8-a179-fefebbcb3e1c:train:0/sampler_weights/router-v4-qwen3.5-4b-lora-r16-epoch2
- Sealed sets: not read, not scored. Sealed scoring waits for the owner's ticks on the 14 change-request candidates and a new order.
- Epoch choice: none made for sealed use. By val macro, epoch 2 (0.6389) is ahead of epoch 1 (0.5355). v4 was superseded by the plan to train v5 on main with the pr5 data.

## Val (n=848)

| | epoch 1 | epoch 2 |
| --- | ---: | ---: |
| macro recall (labels with n>0: 18) | 0.5355 | 0.6389 |
| accuracy | 0.8149 | 0.8750 |
| val gate (beats majority, majority fails) | True | True |
| unsure/off-list | 0 | 0 |

Majority baseline: drawing, accuracy 0.7689, macro 0.0556.

| label | val n | epoch 1 recall | epoch 2 recall |
| --- | ---: | ---: | ---: |
| RFI | 7 | 0.714 | 0.000 |
| submittal | 4 | 0.750 | 1.000 |
| transmittal | 1 | 0.000 | 1.000 |
| VO/CO | 0 | - | - |
| change request | 0 | - | - |
| claim | 0 | - | - |
| payment application | 1 | 1.000 | 0.000 |
| payment certificate | 1 | 1.000 | 1.000 |
| drawing | 652 | 0.971 | 0.985 |
| specification | 3 | 0.000 | 0.667 |
| schedule | 8 | 1.000 | 1.000 |
| report | 44 | 0.000 | 0.659 |
| correspondence | 7 | 0.000 | 0.000 |
| BOQ | 2 | 0.500 | 0.500 |
| contract | 11 | 0.545 | 0.455 |
| method statement | 3 | 1.000 | 1.000 |
| meeting minutes | 10 | 1.000 | 1.000 |
| NCR | 0 | - | - |
| inspection request | 7 | 0.000 | 0.143 |
| site instruction | 8 | 1.000 | 0.750 |
| register/log | 3 | 0.000 | 1.000 |
| other | 76 | 0.158 | 0.342 |

Val has n=0 for VO/CO, change request, claim and NCR, so those classes cannot be judged here. Do not compare this macro directly with v3's 0.687: that figure was measured over 19 labels on v8 labels.
