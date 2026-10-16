# Router run v2 — 2026-10-06 (Dubai, UTC+4)

Teacher: Qwen/Qwen3-8B on Tinker, temperature 0, prompt `_PROMPT` from scripts/router/teacher_label.py (PR #27 branch 542305b), content only (first_512_tokens).
Pool: 6580 input -> 140 empty, 813 duplicate doc_id -> 5627 labelled. Kept (on-list label, confidence >= 0.7): 4555. Excluded 1072 = 60 off-list labels mapped to unsure (teacher never said "unsure"; 0 unparseable) + 1012 below 0.7 (753 of them "other", mostly confidence 0).

Split v2 (driver/split_v2.py): held out by group, stratified. The group key is a document series taken from the filename (stand-in for the folder, which the pool doesn't have). Groups are merged when rows share a filename, an excerpt or a sha. project_id is not part of the key. Seed 20261006. 537 groups: 27 in val, 0 in both.
Train 1334 (3872 before cap; drawing capped from 3138 to 600, seed 20261006, train only). Val 683 (15.0%), not changed.
Sealed isolation: router.isolation.require_isolated (sealed file empty at split time, so 0 excluded). The sealed set was scored once later; see below.

Model: Qwen/Qwen3.5-4B LoRA r16 (alpha 16), 2 epochs, lr 4.73e-4, batch 16, seed 0, 168 steps. max_sequence_length raised from 2048 to 3072 for this run only.
Tinker training run: b0857111-57c4-56f6-bf9b-ed3f8021826b:train:0
Checkpoint (epoch 2): tinker://b0857111-57c4-56f6-bf9b-ed3f8021826b:train:0/sampler_weights/router-qwen3.5-4b-lora-r16-epoch2
Epoch 1: tinker://b0857111-57c4-56f6-bf9b-ed3f8021826b:train:0/sampler_weights/router-qwen3.5-4b-lora-r16-epoch1

Val per-label recall (val n):     epoch1 / epoch2
RFI (18)                 0.50 / 0.50
submittal (4)            1.00 / 0.75
transmittal (3)          0.00 / 1.00
VO/CO (1)                0.00 / 1.00
claim (1)                0.00 / 0.00   (epoch 2 predicts claim 11 times, 0 correct)
payment certificate (2)  1.00 / 1.00
drawing (554)            0.984 / 0.971
spec (29)                0.241 / 0.759
schedule (16)            0.50 / 0.375
report (37)              0.973 / 0.946
correspondence (18)      0.389 / 0.000
other (0)                n/a
Macro recall over 11 labels: 0.508 / 0.664. Accuracy: 0.905 / 0.906.
Majority baseline (always "drawing"): accuracy 0.811, macro 0.091, gate FAIL (as it should).
Model gate: PASS both epochs. It is not a constant predictor, and its macro recall beats the baseline.

Walls:
- Tinker does not serve Qwen/Qwen3-4B (400 "not supported"). Used Qwen/Qwen3.5-4B, the closest 4B Tinker serves.
- The pool has no folder path, so the second-level folder can't be recovered. Used document series from the filename instead.
- train_router._encode breaks on transformers 5.x, where apply_chat_template returns a BatchEncoding. Patched in the driver (_encode_compat). PR #27 needs the same fix.
- With the Qwen3.5 tokenizer, 130 train rows went over max_sequence_length 2048 (longest 2996). Raised to 3072 for this run. PR #27's config is still 2048.
- v1 project split (split_projects.py) gave 4533/22. After dedupe, two project ids hold 99.6% of rows and share the same filenames. Replaced by v2.
- Val is small for VO/CO (1), claim (1), payment certificate (2), transmittal (3) and submittal (4), so those recalls are noisy.
- No CloudAgent tool in this session. Nothing was committed to PR #27.
- Human agreement is done (0.875, kappa 0.864). This run trained before that check, by owner order.

## Sealed score (once)

Scored once. Files: `docs/router/runs/v2/sealed/`. Macro recall 0.715. Accuracy 0.672 (43/64). Every constant predictor scores macro 0.083. Condition (a) holds: the model beats all 12 constants. Condition (b) fails.

sealed gate: FAIL (gate defect: condition (b) unsatisfiable as written); router NOT marked passed.

`PRE_RUN_NOTE.txt` logged the defect before any sealed prediction. The majority check fails only the constant equal to the sealed-set majority (`report`). The other 11 constants pass it, so condition (b) cannot be met. The gate definition, the parser, and the scores were not changed after the run. Sealed gate version 2 applies only to a future fresh sealed set. These v5 scores were not recomputed under version 2 or the label-alias parser.

VO/CO recall is 0/5. 3 outputs said 'change order', which the val parser maps to unsure, and they were not remapped.

