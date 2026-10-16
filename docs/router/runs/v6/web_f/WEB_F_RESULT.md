# Router v6 epoch 3 on the v9 web_f addendum (scored once) and the 312-row combined view

Adapter `tinker://93552b55-3c20-5862-8bbf-2320fb04b236:train:0/sampler_weights/router-v6-qwen3.5-4b-lora-r16-epoch3`
(the PR #61 run record). Main `92f8f0a750940bb88d7d4cb6bc0975bda01837a4` rebuilt and tree-hash verified, plus this addendum only.
`PRE_RUN_NOTE.txt` was written at 19:09:44 +04, before `RUN_STARTED_score.json` (19:09:48) and `RUN_STARTED_blind.json` (19:10:09).
Prompt, sampling, parser (`router.train_router._as_label`, v9 exact aliases), gate (`evaluate_sealed_gate_v8`, gate v2 +
MIN_SEALED_PER_LABEL=12) and the blanker (`scripts/router/name_blind.py`, PR #50, sha256 a235de3c…) are unchanged from PR #61.

## Method

- **Sampled once, new set only:** `score_v6_web_f.py score` sampled the 34 web_f rows once, then `blind` sampled the same 34 rows once with
  party/title fields blanked. Each phase refuses a second run. No failed calls, so no resume was needed.
- **Not re-scored:** v5_retick, v8_retick, cr, thin_c and thin_d were scored once in PR #61. They were not sampled again.
- **Combined (312) is an aggregation, not a new score:** `combine_v6_web_f.py` joins v6's recorded per-row predictions from
  `docs/router/runs/v6/sealed/` (278 rows) with the 34 new rows here, in `configs/router/sealed_eval.json` v9_sealed order. It runs the
  same gate and the same flip table. Gold labels are re-read from the sealed files and must equal the gold stored with each prediction.
- Precision is added for reporting only. It is TP divided by the number of rows predicted as that label.

## Scores

| view | n | macro recall | accuracy | gate_v2 | router_passed |
|---|---|---|---|---|---|
| web_f_addendum (new, scored once) | 34 | 0.9182 | 0.8235 | pass | false |
| combined before (278, PR #61) | 278 | 0.7665 | 0.7266 | pass | false |
| combined now (312, aggregation) | 312 | 0.7857 | 0.7372 | pass | false |

`router_passed` stays false because four labels are still below n=12 on the combined view: claim 9, payment application 6,
site instruction 9 and inspection request 3.

On web_f, 28 of 34 rows were right. The 6 misses were 5 payment certificate rows predicted as payment application, and 1 claim predicted as change request.

## Per class on the combined view (v6)

| class | n before (278) | recall before | n now (312) | recall now | precision now | passable now |
|---|---|---|---|---|---|---|
| RFI | 9 | 1.000 | 12 | 1.000 | 1.000 | yes |
| submittal | 18 | 0.889 | 18 | 0.889 | 0.842 | yes |
| transmittal | 13 | 1.000 | 13 | 1.000 | 1.000 | yes |
| VO/CO | 16 | 0.938 | 16 | 0.938 | 0.938 | yes |
| change request | 14 | 1.000 | 14 | 1.000 | 0.636 | yes |
| claim | 4 | 1.000 | 9 | 0.889 | 0.800 | no |
| payment application | 4 | 0.750 | 6 | 0.833 | 0.357 | no |
| payment certificate | 4 | 0.250 | 15 | 0.467 | 0.700 | yes |
| drawing | 18 | 0.778 | 18 | 0.778 | 0.519 | yes |
| specification | 18 | 0.667 | 18 | 0.667 | 1.000 | yes |
| schedule | 19 | 1.000 | 19 | 1.000 | 0.826 | yes |
| report | 23 | 0.435 | 23 | 0.435 | 1.000 | yes |
| correspondence | 19 | 0.105 | 19 | 0.105 | 0.667 | yes |
| BOQ | 12 | 0.750 | 12 | 0.750 | 0.900 | yes |
| contract | 12 | 0.917 | 12 | 0.917 | 0.846 | yes |
| method statement | 14 | 1.000 | 14 | 1.000 | 1.000 | yes |
| meeting minutes | 9 | 1.000 | 14 | 1.000 | 0.875 | yes |
| NCR | 8 | 1.000 | 12 | 1.000 | 0.800 | yes |
| inspection request | 0 | — | 3 | 1.000 | 0.500 | no |
| site instruction | 8 | 1.000 | 9 | 1.000 | 0.346 | no |
| register/log | 15 | 0.333 | 15 | 0.333 | 1.000 | yes |
| other | 21 | 0.286 | 21 | 0.286 | 0.462 | yes |

VO/CO: combined recall 15/16 = 0.938. The addendum added no VO/CO rows, so this is unchanged.

## Name-blind (PR #50 blanker)

| view | rows with a blanked field | flips | on passable classes | gate |
|---|---|---|---|---|
| web_f_addendum | 23/34 | 0 | 0 | PASS |
| combined (312, aggregation) | — | 6 | 6 | FAIL (correspondence, drawing, other, report, submittal) |

The 6 combined flips are the same 6 rows already recorded in PR #61: 4 in v5_retick and 2 in v8_retick. web_f adds none.
The other per-set name-blind results are unchanged from PR #61: v5_retick PASS, v8_retick FAIL (correspondence, drawing), cr PASS,
thin_c PASS, thin_d PASS.

## Files

- `PRE_RUN_NOTE.txt`, `RUN_STARTED_*.json`, `score.log`: the run trail. The log opens with the driver's dry check.
- `sealed_predictions.jsonl`, `name_blind_predictions.jsonl`, `*_partial.jsonl`: per-row raw output and parsed label for the 34 rows.
  The two prediction files are byte-identical. No row's raw output changed under blanking, and both files store the same per-row count of
  blankable fields (no text is stored). The partial files differ only in completion order.
- `sealed_scores.json`, `name_blind.json`: the web_f view.
- `combined_scores.json`, `combined_name_blind.json`: the 312-row aggregation, including the flip list.
- `score_v6_web_f.py`, `combine_v6_web_f.py`, `inputs.json`: the drivers and input hashes. No document text is stored.
