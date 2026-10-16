# Router v7b run record (repair attempt 1 of 2, the last allowed)

Written after the single sealed scoring. There was no pre-run note.

- Tinker run: `467096c8-5c73-595c-b5a8-e92039dba560:train:0`
- Chosen checkpoint (epoch 1, on val macro recall over present classes; early stop with patience 1; accuracy is not used to select):
  `tinker://467096c8-5c73-595c-b5a8-e92039dba560:train:0/sampler_weights/router-v7b-qwen3.5-4b-lora-r16-epoch1`
- Val macro recall present classes: e1 0.6583, e2 0.5862 (accuracy, report only: 0.817 / 0.809)
- Data commit `b3e1001` (PR #66, merged after PR #67 `9747f70`). Training tree `4f5fa14` equals the merged tree, all 525 blobs. Seed 0.
- Recipe: v6/v7 unchanged (Qwen/Qwen3.5-4B, LoRA r16/alpha 16, lr 4.73e-4, batch 16, max_seq 6656, max 3 epochs, patience 1).
- Sampler: tempered inverse frequency loss weight, exponent 0.5, cap 3x natural share. Weights are read from
  `data/router/v7b/class_weights.json` (sha256 2b6b57a4...). They are not in code.
- Pairs (`configs/router/pairs_v7b.json`, `docs/router/V7B_TRAINING.md`): contrastive 797 (content-named neighbour first, 3x edge
  cap, project round-robin), combined-form 0, catch-all 183, catch-all sibling 183.
- Train 2882 examples (1719 base + 1163 pairs), val 901. Preflight: 0 rows over max_seq.
- Scorer, parser, prompt (system sha256 b1ac9472...), blanker and gate are the same as v7. `sealed/score_v7.py` and
  `sealed/compare_v7.py` are byte-identical to the v7 record copies.

## Sealed result (312 rows, scored once, epoch 1)

| set | n | macro v7 | macro v7b | acc v7 | acc v7b |
|---|---|---|---|---|---|
| v5_retick | 64 | 0.7482 | 0.6238 | 0.7031 | 0.5625 |
| v8_retick | 185 | 0.7730 | 0.7576 | 0.7622 | 0.7297 |
| cr_addendum | 14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| thin_c_addendum | 2 | 0.5000 | 1.0000 | 0.5000 | 1.0000 |
| thin_d_addendum | 13 | 0.6389 | 0.5000 | 0.7692 | 0.6923 |
| web_f_addendum | 34 | 0.8131 | 0.9006 | 0.7059 | 0.7941 |
| combined | 312 | 0.7843 | 0.7517 | 0.7532 | 0.7147 |

v6 combined: macro 0.7857, accuracy 0.7372. v7b gate_v2 passed; router gate (incl. name-blind) not passed.

## Per class v6 -> v7 -> v7b (combined 312)

| label | n | recall v6 | recall v7 | recall v7b | precision v6 | precision v7 | precision v7b | predicted v6 / v7 / v7b |
|---|---|---|---|---|---|---|---|---|
| RFI | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 12 / 12 / 12 |
| submittal | 18 | 0.889 | 0.889 | 0.556 | 0.842 | 0.762 | 0.769 | 19 / 21 / 13 |
| transmittal | 13 | 1.000 | 0.923 | 0.538 | 1.000 | 1.000 | 1.000 | 13 / 12 / 7 |
| VO/CO | 16 | 0.938 | 0.875 | 0.875 | 0.938 | 0.933 | 0.933 | 16 / 15 / 15 |
| change request | 14 | 1.000 | 1.000 | 1.000 | 0.636 | 0.933 | 0.700 | 22 / 15 / 20 |
| claim | 9 | 0.889 | 1.000 | 1.000 | 0.800 | 0.818 | 0.375 | 10 / 11 / 24 |
| payment application | 6 | 0.833 | 1.000 | 1.000 | 0.357 | 0.286 | 0.286 | 14 / 21 / 21 |
| payment certificate | 15 | 0.467 | 0.400 | 0.400 | 0.700 | 0.750 | 1.000 | 10 / 8 / 6 |
| drawing | 18 | 0.778 | 1.000 | 1.000 | 0.519 | 0.621 | 0.545 | 27 / 29 / 33 |
| specification | 18 | 0.667 | 0.778 | 0.722 | 1.000 | 0.933 | 1.000 | 12 / 15 / 13 |
| schedule | 19 | 1.000 | 1.000 | 0.947 | 0.826 | 0.704 | 0.857 | 23 / 27 / 21 |
| report | 23 | 0.435 | 0.913 | 0.696 | 1.000 | 0.724 | 0.941 | 10 / 29 / 17 |
| correspondence | 19 | 0.105 | 0.158 | 0.000 | 0.667 | 1.000 | — | 3 / 3 / 0 |
| BOQ | 12 | 0.750 | 0.750 | 1.000 | 0.900 | 0.818 | 0.800 | 10 / 11 / 15 |
| contract | 12 | 0.917 | 1.000 | 0.917 | 0.846 | 0.600 | 1.000 | 13 / 20 / 11 |
| method statement | 14 | 1.000 | 0.929 | 1.000 | 1.000 | 1.000 | 1.000 | 14 / 13 / 14 |
| meeting minutes | 14 | 1.000 | 0.857 | 1.000 | 0.875 | 0.923 | 1.000 | 16 / 13 / 14 |
| NCR | 12 | 1.000 | 0.500 | 0.833 | 0.800 | 1.000 | 0.769 | 15 / 6 / 13 |
| inspection request | 3 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.429 | 6 / 6 / 7 |
| site instruction | 9 | 1.000 | 0.778 | 0.444 | 0.346 | 0.875 | 1.000 | 26 / 8 / 4 |
| register/log | 15 | 0.333 | 0.267 | 0.133 | 1.000 | 0.800 | 1.000 | 5 / 5 / 2 |
| other | 21 | 0.286 | 0.238 | 0.476 | 0.462 | 0.417 | 0.333 | 13 / 12 / 30 |

"—" = never predicted. Precision rule: for "no class precision < 0.5": every label that is predicted at least once or present in gold on the 312 rows is
checked. precision = TP / rows predicted as that label. A label present in gold but never predicted counts as precision 0 (fails).
'unsure' is not a label and is not checked.

## Acceptance (owner, Oct 8 20:09)

| criterion | v7 | v7b | v7b result |
|---|---|---|---|
| correspondence recall >= 0.5 | 0.158 | 0.000 | FAIL |
| payment certificate recall >= 0.75 | 0.400 | 0.400 | FAIL |
| no class precision < 0.5 (labels predicted or present; never-predicted present label = 0) | payment application 0.286, other 0.417 | claim 0.375, payment application 0.286, correspondence 0.000, inspection request 0.429, other 0.333 | FAIL |
| VO/CO >= 15/16 | 14/16 | 14/16 | FAIL |
| change request precision >= 0.75 | 0.933 | 0.700 | FAIL |

**Not promoted. 0 of 5 criteria met.** v6 stays the serving router.

## Name-blind (same checkpoint, same parser, filenames/titles blanked)

| set | flips v6 | flips v7 | flips v7b | v7b gate |
|---|---|---|---|---|
| v5_retick | 4 | 1 | 3 | pass |
| v8_retick | 2 | 1 | 3 | FAIL correspondence, submittal |
| cr_addendum | 0 | 0 | 0 | pass |
| thin_c_addendum | 0 | 0 | 0 | pass |
| thin_d_addendum | 0 | 1 | 0 | pass |
| web_f_addendum | 0 | 0 | 0 | pass |
| combined | 6 | 3 | 6 | FAIL correspondence, other, submittal |

The six v7b combined flips. Two go correct -> wrong: a submittal row (submittal -> other) and an other row (other -> inspection
request). Four go wrong -> wrong: two to other, one other -> drawing, one claim -> other. Blanking pushes the adapter toward `other`.
The combined name-blind gate fails on correspondence, other and submittal.

## Diagnosis (sealed confusion matrix + val; no sealed text read)

v7b chose epoch 1. Val macro recall over present classes was 0.6583 at epoch 1 and fell to 0.5862 at epoch 2, so patience 1 stopped the run.
v7 trained 3 epochs. The epoch-1 val gain came from tiny val classes (NCR n=3 0.33 -> 1.0, inspection request n=7 0.29 -> 1.0). On the same val,
correspondence went 0.10 -> 0.00, site instruction 0.875 -> 0.00 and report 0.23 -> 0.02. The rule picks epochs on macro recall and never on accuracy
or sealed results, so it worked as designed. The val set is too thin in the action and catch-all classes to separate these adapters.

What v7b changed on sealed versus v7:
- Catch-all sibling pairs moved mass onto `other`: predicted 12 -> 30, recall 0.238 -> 0.476, precision 0.417 -> 0.333.
  transmittal (4) and submittal (2) now leak into `other`.
- claim is over-predicted: 11 -> 24 predictions, precision 0.818 -> 0.375. 7 correspondence rows went to claim.
- correspondence 3/19 -> 0/19. It is never predicted. Misses: other 7, claim 7, payment application 3, change request 1.
- register/log 4/15 -> 2/15. 10 went to drawing. The named-neighbour rule found only 16 register rows that name a neighbour, and the
  drawing edge got 4 rows.
- Unchanged: payment certificate 6/15 (9 -> payment application), VO/CO 14/16.

Correspondence: the label collapses in every run (v6 0.105, v7 0.158, v7b 0.000 on sealed; v7 0.10 and v7b 0.00 on val). Sealed
correspondence rows go to the label of their subject matter (claim, payment application, change request) or to other. The 80 training
correspondence rows are a weak class signal: 52 of them come from one project, and many are not letters or emails between parties
(placeholder templates, site quick-notes, tender clauses, drawing annotations). So the model never learns that a letter about a
claim or a payment is correspondence. Loss weights and restricted-menu pairs only reweight these same rows, and three runs show
that is not enough. Correspondence needs clean letter/email training rows from many projects, including letters whose subject is an
instrument topic (claims, payments, variations, instructions), labelled from content under the v9 definition.

Payment certificate: train has no certificate-side rows (certified amount AND certifier signature/date present). The
combined-form rule (1c) therefore produced 0 pairs in v7 and v7b, and 9 of 15 sealed certificates are read as payment
application in both. No generic sampler or pair change can create the missing side. Payment certificate needs certificate-side
training rows.

## Park statement

Repair budget used (v7, v7b; attempt 2 of 2). **The router is parked; no v7c will be trained.** v6 remains the serving router.
Unparking needs new training data, not mechanism changes:
1. payment certificate: certificate-side rows (completed certifier section with a certified amount and certifier signature/date),
   ideally from projects that also contribute application-side rows so the combined-form pairs can form;
2. correspondence: clean, multi-project letter/email rows, including letters about instrument topics, labelled from content.
After such data merges, a new run must be trained once and scored once under the same acceptance criteria.
