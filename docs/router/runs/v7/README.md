# Router v7 run record

Written after the single sealed scoring. There was no pre-run note.

- Tinker run: `06e01fff-72b3-585a-ac27-ea306c30443e:train:0`
- Chosen checkpoint (epoch 3, on val macro recall over present classes; accuracy is not used to select):
  `tinker://06e01fff-72b3-585a-ac27-ea306c30443e:train:0/sampler_weights/router-v7-qwen3.5-4b-lora-r16-epoch3`
- Val macro recall present classes: e1 0.5367, e2 0.5926, e3 0.6501 (accuracy, report only: 0.810 / 0.827 / 0.839)
- Data commit `1f9106c` (PR #64), training tree `73e5d4f` (equals the merged tree, all 467 blobs). Seed 0.
- Recipe: v6 unchanged (Qwen/Qwen3.5-4B, LoRA r16/alpha 16, lr 4.73e-4, batch 16, max_seq 6656, max 3 epochs, patience 1).
- Sampler: tempered inverse frequency loss weight, exponent 0.5, cap 3x natural share. Weights are read from
  `data/router/v7/class_weights.json` (sha256 3ca84854...). They are not in code.
- Pairs: contrastive 797, combined-form 0, catch-all 183 (`data/router/v7/`).
- Train 2699 examples (1719 base + 980 pairs), val 901. Preflight: 0 rows over max_seq.

Files: `RUN_RECORD.json`, `metrics.json`, `checkpoints.json`, `epoch_choice.json`, `preflight.*`, `train-v7.log` (env line reduced to a name
count), `val_predictions_epoch{1,2,3}.jsonl`, `driver/`, `sealed/` (scorer, comparer, box-side source map, inputs, predictions, scores,
name-blind, v6 comparison).

## Sealed result (312 rows, scored once, epoch 3)

| set | n | macro recall | accuracy |
|---|---|---|---|
| v5_retick | 64 | 0.7482 | 0.7031 |
| v8_retick | 185 | 0.7730 | 0.7622 |
| cr_addendum | 14 | 1.0000 | 1.0000 |
| thin_c_addendum | 2 | 0.5000 | 0.5000 |
| thin_d_addendum | 13 | 0.6389 | 0.7692 |
| web_f_addendum | 34 | 0.8131 | 0.7059 |
| combined | 312 | 0.7843 | 0.7532 |

v6 combined: macro 0.7857, accuracy 0.7372. gate_v2 passed; router gate (incl. name-blind) not passed.

## Per class v6 -> v7 (combined 312)

| label | n | recall v6 | recall v7 | precision v6 | precision v7 | predicted v6 | predicted v7 |
|---|---|---|---|---|---|---|---|
| RFI | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 12 | 12 |
| submittal | 18 | 0.889 | 0.889 | 0.842 | 0.762 | 19 | 21 |
| transmittal | 13 | 1.000 | 0.923 | 1.000 | 1.000 | 13 | 12 |
| VO/CO | 16 | 0.938 | 0.875 | 0.938 | 0.933 | 16 | 15 |
| change request | 14 | 1.000 | 1.000 | 0.636 | 0.933 | 22 | 15 |
| claim | 9 | 0.889 | 1.000 | 0.800 | 0.818 | 10 | 11 |
| payment application | 6 | 0.833 | 1.000 | 0.357 | 0.286 | 14 | 21 |
| payment certificate | 15 | 0.467 | 0.400 | 0.700 | 0.750 | 10 | 8 |
| drawing | 18 | 0.778 | 1.000 | 0.519 | 0.621 | 27 | 29 |
| specification | 18 | 0.667 | 0.778 | 1.000 | 0.933 | 12 | 15 |
| schedule | 19 | 1.000 | 1.000 | 0.826 | 0.704 | 23 | 27 |
| report | 23 | 0.435 | 0.913 | 1.000 | 0.724 | 10 | 29 |
| correspondence | 19 | 0.105 | 0.158 | 0.667 | 1.000 | 3 | 3 |
| BOQ | 12 | 0.750 | 0.750 | 0.900 | 0.818 | 10 | 11 |
| contract | 12 | 0.917 | 1.000 | 0.846 | 0.600 | 13 | 20 |
| method statement | 14 | 1.000 | 0.929 | 1.000 | 1.000 | 14 | 13 |
| meeting minutes | 14 | 1.000 | 0.857 | 0.875 | 0.923 | 16 | 13 |
| NCR | 12 | 1.000 | 0.500 | 0.800 | 1.000 | 15 | 6 |
| inspection request | 3 | 1.000 | 1.000 | 0.500 | 0.500 | 6 | 6 |
| site instruction | 9 | 1.000 | 0.778 | 0.346 | 0.875 | 26 | 8 |
| register/log | 15 | 0.333 | 0.267 | 1.000 | 0.800 | 5 | 5 |
| other | 21 | 0.286 | 0.238 | 0.462 | 0.417 | 13 | 12 |

Precision rule: for "no class precision < 0.5": every label that is predicted at least once or present in gold on the 312 rows is
checked. precision = TP / rows predicted as that label. A label present in gold but never predicted counts as precision 0 (fails).
'unsure' is not a label and is not checked.

## Acceptance (owner, Oct 8 20:09)

| criterion | v7 | result |
|---|---|---|
| correspondence recall >= 0.5 | 0.158 | FAIL |
| payment certificate recall >= 0.75 | 0.4 | FAIL |
| no class precision < 0.5 (labels predicted or present; never-predicted present label = 0) | {'payment application': 0.2857142857142857, 'other': 0.4166666666666667} | FAIL |
| VO/CO >= 15/16 | 14/16 | FAIL |
| change request precision >= 0.75 | 0.933 | PASS |

**Not promoted.** v6 stays the serving router.

## Name-blind (same checkpoint, same parser, filenames/titles blanked)

| set | flips v6 | flips v7 | gate |
|---|---|---|---|
| v5_retick | 4 | 1 | pass |
| v8_retick | 2 | 1 | FAIL other |
| cr_addendum | 0 | 0 | pass |
| thin_c_addendum | 0 | 0 | pass |
| thin_d_addendum | 0 | 1 | pass |
| web_f_addendum | 0 | 0 | pass |
| combined | 6 | 3 | FAIL NCR, other, submittal |

Combined flips (3): one submittal row went right -> wrong (submittal -> transmittal, 2 fields blanked); one `other` row went wrong -> right
(payment certificate -> other, 4 fields blanked); one NCR row went right -> wrong with 0 fields blanked, i.e. sampler nondeterminism at
temperature 0, not a name dependency.

## Diagnosis (from the sealed confusion matrix and the val epoch-3 confusion; no sealed text read)

Fixed relative to v6: over-prediction of the thin action labels (site instruction precision 0.346 -> 0.875, change request 0.636 -> 0.933),
report recall 0.435 -> 0.913, drawing recall 0.778 -> 1.0, combined accuracy 0.737 -> 0.753, name-blind flips 6 -> 3.

Still failing:
- correspondence 3/19. The label is predicted only 3 times. Misses go to other 5, contract 3, report 3, payment application 3. Val shows
  the same collapse (correspondence recall 0.10 at epoch 3), so the training mix causes it, not a sealed shift. In v7 the catch-all pairs (1d)
  set each catch-all against the instrument labels only. correspondence vs other / report / register-log was never practised in a restricted menu.
  80 training correspondence rows; 52 of them come from one project.
- payment certificate 6/15, 9 -> payment application. Data gap: the train split has no certificate-side rows that the combined-form
  rule (1c) can pair, so 1c produced 0 pairs. No generic loss or pair change can supply certificate-side examples. This criterion
  stays unreachable until signed or certified payment certificates enter train.
- payment application precision 0.286: 21 predictions = 6 correct + 9 payment certificate + 3 correspondence + 3 other.
- other precision 0.417: 12 predictions = 5 correct + 5 correspondence + 2 NCR (catch-all vs catch-all again).
- register/log 4/15, 9 -> drawing. register/log has 14 Is-not neighbours, so its 48-row contrastive budget gave about 3 rows per edge. The
  neighbour that the row's own text names (here drawing) got no more practice than any other.
- VO/CO 14/16: 1 -> site instruction, 1 -> schedule.

Generic repair for v7b (repair attempt 1 of 2; own bundle, own run record):
1. Contrastive (1b) neighbour order is content-driven. Is-not neighbours that are named in the row's own text head come first, and rows that
   name a neighbour fill the per-class budget first. Round-robin stays as the fallback.
2. Catch-all sibling pairs: each catch-all row also gets a two-label menu against one sibling catch-all. A sibling named in its head comes first;
   otherwise round-robin.
3. Eligible rows are ordered round-robin across projects, so one project cannot fill a class budget.
The sampler rule (exponent 0.5, cap 3x) and the training recipe are unchanged.
