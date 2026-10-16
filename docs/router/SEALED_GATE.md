# Sealed router gate

Version 2 is the active rule. It is registered for the next fresh sealed set. It does not rescore `data/eval/router_unseen/`. `sealed_scoring` is false. The v5 sealed set is spent. It was scored once under version 1, and that recorded result stays:

sealed gate: FAIL (gate defect: condition (b) unsatisfiable as written); router NOT marked passed.

The router is not marked passed. Version 2 applies only to a future fresh sealed set. The v5 predictions are not re-graded under version 2, and they are not re-parsed with the label-alias parser. This file is not a score.

The val gate in `scripts/router/majority_gate.py` and `scripts/router/val_metrics.py` is unchanged. This gate is separate and applies to sealed rows of a future set.

The machine-readable field is `sealed_gate` in `configs/router/sealed_eval.json`. `sealed_gate.version` is 2. `sealed_gate.applies_to` is `future_fresh_sealed_set`.

## Version 2

The model passes only if both checks below are true. Any other result is a fail. A tie does not pass.

The same rule is `RULE` in `scripts/router/sealed_gate.py`:

Version 2. The model passes only if macro recall beats every constant single-label predictor on the sealed rows, and each of those constant predictors fails the existing majority-gate thresholds on its own. Each constant is evaluated per label, not only when that label is the set majority. Macro recall is the mean of per-label recall over labels present in the sealed rows. The constant predictors include always-drawing, the training majority, and always-report, the sealed-set majority. A tie does not beat a constant predictor. The majority gate is scripts/router/majority_gate.py with majority_class_fail left unchanged. The val gate is unchanged. sealed_scoring stays false. Version 2 applies only to a future fresh sealed set. The v5 sealed set stays FAIL under version 1 and is not re-scored.

1. Macro recall on the sealed rows is strictly greater than the macro recall of every constant single-label predictor. A constant single-label predictor emits one declared label on every sealed row. That set includes always-`drawing` (the training majority) and always-`report` (the sealed-set majority on v5). The comparison covers every declared label.
2. Each of those constant predictors, on its own, fails the existing majority-gate thresholds. The check is per constant, and it covers a label that is not the set majority. For each declared label L, `majority_gate.evaluate` runs on a row set whose gold labels are all L and whose predictions are all L, with `majority_class_fail` left as configured (`true` in `configs/router/sealed_eval.json`). That is the same failure the val gate uses: every prediction is the majority label, including when accuracy is 1.

Macro recall is the mean of per-label recall over labels present in the sealed rows. A label with no sealed row is left out of the mean. Per-label recall is `router.val_metrics.per_label_recall`.

The sealed labels are the owner's ticks. This gate does not read a proposal file and does not change a label.

## Version 1 (history)

Version 1 was the rule when the v5 sealed set was scored. The result of that score stays FAIL. The text below is the version 1 rule, kept as history. The active rule is version 2 above.

The model passes only if macro recall beats every constant single-label predictor on the sealed rows, and each of those constant predictors fails the existing majority gate. Macro recall is the mean of per-label recall over labels present in the sealed rows. The constant predictors include always-drawing, the training majority, and always-report, the sealed-set majority. A tie does not beat a constant predictor. The majority gate is scripts/router/majority_gate.py with majority_class_fail left unchanged. sealed_scoring stays false.

Version 1 called `majority_gate.evaluate` on the real sealed gold for each constant predictor. That function fails a predictor when every prediction is the majority label of those rows. The set-majority constant is the one that fails. On v5 that label is `report`. The other eleven constants pass that call, so condition (b) cannot be met on a set with more than one label. The recorded v5 result is that gate defect. Version 2 leaves the recorded result in place.

## Version 2 + per-label n rule (v8)

`evaluate_sealed_gate_v8` in `scripts/router/sealed_gate.py` keeps version-2 logic and the `RULE` text unchanged. It adds `MIN_SEALED_PER_LABEL = 12`.

For each declared label the report is `{"label", "n", "recall", "passable"}`. `n` is the number of sealed gold documents of that label. `recall` is null when `n` is 0. `passable` is true only when `n` is at least 12. A label with `n` below 12 is marked `not passable`.

`router_passed` is true only when `gate_v2["passed"]` is true and every declared label has `n` of at least 12. Nothing else is loosened. The v5 sealed set is not re-scored under this rule.

## v9 sealed sets and the change-request addendum (pr9)

`configs/router/sealed_eval.json` → `v9_sealed` lists the v9 sealed sets: `v9_retick/v5` (64), `v9_retick/v8` (185), and the new owner-ticked change-request addendum `v9_cr_addendum` (14). Together they make 263 rows. The addendum is a new file; no existing sealed row was swapped. Change request has 14 sealed rows in the combined set, so it is passable at `MIN_SEALED_PER_LABEL = 12`. The gate (`evaluate_sealed_gate_v8`) and the parser are unchanged. Each set is scored at most once per adapter. The combined view is computed from those single predictions and is never re-sampled.

v9 web_f addendum (2026-10-08): `v9_web_f_addendum` adds 34 owner-ticked rows as a new sealed file (6 unsure rows held, never sealed), so `v9_sealed` combined is 312 rows. RFI, payment certificate, meeting minutes and NCR reach n >= 12. Claim, payment application, site instruction and inspection request stay below 12. The per-(project, class) cap of 3 is checked against all registered sealed rows, and exhausted pairs are recorded from data in `data/router/sealed_cap_exhaustion.json`. The gate and the parser are unchanged.

v9 web_g addendum (2026-10-09): `v9_web_g_addendum` adds 7 owner-accepted rows as a new sealed file (2 rows held: one duplicate source document, one not accepted under the suggested class). `v9_sealed` combined is 319 rows. Payment application is 10, claim is 10, site instruction is 11, inspection request stays 3. All four remain below n >= 12. The cap record is recomputed from data. The gate and the parser are unchanged.
