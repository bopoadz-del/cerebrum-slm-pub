# Router v3 epoch 1: sealed score under v9 labels, plus the name-blind check

Owner order 2026-10-07 17:03 +04, steps 3 and 4. Each set was scored ONCE, on main 868e5af (PR #41 merged). PRE_RUN_NOTE.txt was written before any sealed prediction.

- **Adapter:** `tinker://34e98763-665a-534a-906f-651cc7e8c6f3:train:0/sampler_weights/router-v3-qwen3.5-4b-lora-r16-epoch1` (Qwen/Qwen3.5-4B LoRA r16).
- **Why epoch 1:** it was chosen before any sealed score, because its val macro recall was higher (0.687 vs 0.635 for epoch 2).
- **Prompt:** `prompt_for(labels)`, using the v9 22 labels from `configs/router/labels.json`.
- **Input:** `content_text(candidate)`.
- **Sampling:** temperature 0, max_tokens 32, seed 0, no thinking.
- **Parser:** `parse_router_output` (v9 exact aliases).
- **Gate:** `evaluate_sealed_gate_v8` (gate v2 + MIN_SEALED_PER_LABEL=12), unchanged.
- **Training labels:** the adapter was trained on the v8 21-label prompt, so "change request" was never a training target.

## v5 re-ticked (64 rows)

**router_passed: False**. gate_v2.passed: True (beats every constant: True; every constant fails majority gate: True).
Macro recall 0.6399; accuracy 0.6094; unsure/off-list outputs 1.

| label | n | recall | passable |
|---|---:|---:|---|
| RFI | 5 | 1.000 | not passable (n<12) |
| submittal | 5 | 1.000 | not passable (n<12) |
| transmittal | 5 | 1.000 | not passable (n<12) |
| VO/CO | 5 | 0.200 | not passable (n<12) |
| change request | 0 | n/a | not passable (n<12) |
| claim | 2 | 0.500 | not passable (n<12) |
| payment application | 2 | 1.000 | not passable (n<12) |
| payment certificate | 3 | 0.333 | not passable (n<12) |
| drawing | 5 | 1.000 | not passable (n<12) |
| specification | 6 | 0.500 | not passable (n<12) |
| schedule | 5 | 1.000 | not passable (n<12) |
| report | 8 | 0.500 | not passable (n<12) |
| correspondence | 6 | 0.000 | not passable (n<12) |
| BOQ | 0 | n/a | not passable (n<12) |
| contract | 0 | n/a | not passable (n<12) |
| method statement | 0 | n/a | not passable (n<12) |
| meeting minutes | 0 | n/a | not passable (n<12) |
| NCR | 0 | n/a | not passable (n<12) |
| inspection request | 0 | n/a | not passable (n<12) |
| site instruction | 0 | n/a | not passable (n<12) |
| register/log | 0 | n/a | not passable (n<12) |
| other | 7 | 0.286 | not passable (n<12) |

## v8 re-ticked (185 rows)

**router_passed: False**. gate_v2.passed: True (beats every constant: True; every constant fails majority gate: True).
Macro recall 0.6585; accuracy 0.7405; unsure/off-list outputs 0.

| label | n | recall | passable |
|---|---:|---:|---|
| RFI | 4 | 1.000 | not passable (n<12) |
| submittal | 13 | 0.923 | yes |
| transmittal | 8 | 1.000 | not passable (n<12) |
| VO/CO | 11 | 0.000 | not passable (n<12) |
| change request | 0 | n/a | not passable (n<12) |
| claim | 2 | 0.000 | not passable (n<12) |
| payment application | 1 | 0.000 | not passable (n<12) |
| payment certificate | 0 | n/a | not passable (n<12) |
| drawing | 13 | 1.000 | yes |
| specification | 12 | 0.833 | yes |
| schedule | 14 | 1.000 | yes |
| report | 13 | 0.923 | yes |
| correspondence | 12 | 0.167 | yes |
| BOQ | 12 | 0.667 | yes |
| contract | 12 | 1.000 | yes |
| method statement | 14 | 1.000 | yes |
| meeting minutes | 5 | 1.000 | not passable (n<12) |
| NCR | 2 | 0.000 | not passable (n<12) |
| inspection request | 0 | n/a | not passable (n<12) |
| site instruction | 8 | 0.875 | not passable (n<12) |
| register/log | 15 | 0.267 | yes |
| other | 14 | 0.857 | yes |

Change request has n=0 in both sets, so it is not passable. router_passed cannot be true while any label has n<12.

## Name-blind check (new gate)

The same rows were re-classified with the title-block and party field values blanked by position (`scripts/router/name_blind.py`). The adapter, prompt, sampling and parser were the same.
A flip is a changed prediction. Any flip on a passable class (n>=12 in that set) is a FAIL. The fix is more rows from other projects, never a patch.

### v5: PASS

Rows with at least one blanked field: 41/64; fields blanked: 122; flips: 2; accuracy 0.6094 -> 0.5781.
Failing passable classes: none.

| gold class | n | flips | correct before | correct after |
|---|---:|---:|---:|---:|
| RFI | 5 | 0 | 5 | 5 |
| submittal | 5 | 0 | 5 | 5 |
| transmittal | 5 | 0 | 5 | 5 |
| VO/CO | 5 | 0 | 1 | 1 |
| claim | 2 | 0 | 1 | 1 |
| payment application | 2 | 0 | 2 | 2 |
| payment certificate | 3 | 0 | 1 | 1 |
| drawing | 5 | 0 | 5 | 5 |
| specification | 6 | 0 | 3 | 3 |
| schedule | 5 | 0 | 5 | 5 |
| report | 8 | 1 | 4 | 3 |
| correspondence | 6 | 0 | 0 | 0 |
| other | 7 | 1 | 2 | 1 |

### v8: FAIL

Rows with at least one blanked field: 95/185; fields blanked: 239; flips: 4; accuracy 0.7405 -> 0.7568.
Failing passable classes: BOQ, register/log.

| gold class | n | flips | correct before | correct after |
|---|---:|---:|---:|---:|
| RFI | 4 | 0 | 4 | 4 |
| submittal | 13 | 0 | 12 | 12 |
| transmittal | 8 | 0 | 8 | 8 |
| VO/CO | 11 | 1 | 0 | 1 |
| claim | 2 | 0 | 0 | 0 |
| payment application | 1 | 0 | 0 | 0 |
| drawing | 13 | 0 | 13 | 13 |
| specification | 12 | 0 | 10 | 10 |
| schedule | 14 | 0 | 14 | 14 |
| report | 13 | 0 | 12 | 12 |
| correspondence | 12 | 0 | 2 | 2 |
| BOQ | 12 | 1 | 8 | 9 |
| contract | 12 | 0 | 12 | 12 |
| method statement | 14 | 0 | 14 | 14 |
| meeting minutes | 5 | 0 | 5 | 5 |
| NCR | 2 | 0 | 0 | 0 |
| site instruction | 8 | 1 | 7 | 8 |
| register/log | 15 | 1 | 4 | 4 |
| other | 14 | 0 | 12 | 12 |

After this run, v5 and v8 are spent for this adapter. Neither may be re-scored or tuned on, and no sealed document or label goes into training.
