# Router v7 training mechanism and data (owner task 2026-10-08 20:09 +04)

Status: mechanism and generated data only. v7 is trained once on this tree; its run record goes in
`docs/router/runs/v7/` and is written after scoring. There is no pre-run note.

## Why (owner's read of v6, binding)
v6 over-predicts the thin action classes (precision on the 312 combined sealed rows: site instruction 0.346, payment
application 0.357, inspection request 0.500, change request 0.636) and under-predicts the catch-alls (recall: correspondence
0.105, other 0.286, register/log 0.333, report 0.435). The cause is v6's full inverse-frequency balancing (PR #45), which
gave every class the same 4.6% of the gradient. Payment certificate recall 0.467 is a data gap: v6 never saw signed certifier
blocks.

## 1a. Tempered class weights, stored as data
`scripts/router/balance_v7.py`, config `configs/router/train_v7.json`, weights file `data/router/v7/class_weights.json`.

- N = v7 training examples (base train rows plus generated pairs), K = classes present, n_c = examples of class c.
- raw_c = (N / (K · n_c)) ^ 0.5, the square root of v6's inverse frequency.
- w_c = raw_c / mean over examples, so the mean weight is 1.0 and the gradient mass per epoch matches an unweighted run.
- Cap: a class's effective share n_c · w_c / N may be at most 3× its natural share n_c / N, which means w_c ≤ 3. Capped classes are held
  at 3 and the rest are rescaled until it is stable. On the committed examples the cap does not bind; the largest weight is 2.419.
- The weight multiplies the label-token loss weights exactly as in v6. Rows are not duplicated or dropped.
- `router.v7_pairs main` writes the weights file. `router.train_router.fit_router_v7` reads it and refuses it unless it equals a
  recomputation from the label counts of the examples about to be trained. No label name appears in the mechanism or its config.

Effective share of the gradient by class (v6 rule on the current 1719 train rows → v7 file):

| class | v6 rows | v6 weight | v6 share | v7 examples | v7 weight | v7 share |
|---|---|---|---|---|---|---|
| drawing | 400 | 0.200 | 0.046 | 448 | 0.560 | 0.093 |
| other | 300 | 0.266 | 0.046 | 348 | 0.635 | 0.082 |
| report | 123 | 0.649 | 0.046 | 219 | 0.801 | 0.065 |
| contract | 134 | 0.596 | 0.046 | 182 | 0.878 | 0.059 |
| correspondence | 80 | 0.998 | 0.046 | 176 | 0.893 | 0.058 |
| BOQ | 76 | 1.050 | 0.046 | 124 | 1.064 | 0.049 |
| register/log | 43 | 1.856 | 0.046 | 124 | 1.064 | 0.049 |
| specification | 66 | 1.209 | 0.046 | 114 | 1.110 | 0.047 |
| meeting minutes | 58 | 1.376 | 0.046 | 106 | 1.151 | 0.045 |
| schedule | 50 | 1.596 | 0.046 | 98 | 1.197 | 0.043 |
| VO/CO | 48 | 1.663 | 0.046 | 96 | 1.210 | 0.043 |
| change request | 47 | 1.698 | 0.046 | 94 | 1.222 | 0.043 |
| inspection request | 48 | 1.663 | 0.046 | 94 | 1.222 | 0.043 |
| site instruction | 48 | 1.663 | 0.046 | 92 | 1.236 | 0.042 |
| payment application | 37 | 2.157 | 0.046 | 73 | 1.387 | 0.038 |
| RFI | 36 | 2.217 | 0.046 | 63 | 1.493 | 0.035 |
| submittal | 31 | 2.575 | 0.046 | 61 | 1.517 | 0.034 |
| method statement | 27 | 2.956 | 0.046 | 53 | 1.628 | 0.032 |
| claim | 25 | 3.193 | 0.046 | 50 | 1.676 | 0.031 |
| NCR | 18 | 4.434 | 0.046 | 36 | 1.975 | 0.026 |
| payment certificate | 12 | 5.108 | 0.036 | 24 | 2.419 | 0.022 |
| transmittal | 12 | 5.108 | 0.036 | 24 | 2.419 | 0.022 |

## Pair format (fits the single-label constrained-decoding router)
A pair row is the same chat as a training row: system prompt, document text, one-label answer. The difference is a restricted
label menu. The system prompt is `router.teacher_label.prompt_for(menu)`, so each listed label keeps its approved v9 definition,
Is-not line, regional names and notes verbatim, along with the design rules. The answer is the row's own teacher label, which is always on the menu.
The pair puts a document beside one named neighbour, so the loss is spent on that one distinction.
Validation, epoch choice and sealed scoring keep the full 22-label prompt (system sha256 b1ac9472…), the
same sampling settings and the unchanged exact-alias parser. Pair files store references (doc_id, file_sha256, label, menu), not text.
`router.v7_pairs.materialise` rebuilds the chats from `data/router/train.jsonl` and refuses any mismatch. Only train rows that pass the
content gate (`collect.content_gate.gate`: at least 200 chars and a filled field) are used. Val rows are never used.

Generator: `scripts/router/v7_pairs.py`. Rules: `configs/router/pairs_v7.json`. Counts: `data/router/v7/pairs_summary.json`.
Run `PYTHONPATH=scripts python -m router.v7_pairs --root .` to regenerate, or add `--check` to compare against the committed files.

### 1b. Contrastive pairs from the Is-not lines (797 rows)
- A neighbour is a label named in a definition's Is-not line, by label name or by an unambiguous regional name parsed from the
  same file. "A variation order" in an Is-not line names VO/CO, for example. Edges are made symmetric, which gives 60 edges.
- Per-class budget = the median per-class row count of train, which is 48. Each class's eligible rows are ordered by sha256(seed|label|doc_id).
  The first min(eligible, 48) rows each get one neighbour, round-robin over the class's neighbours. "other" has no Is-not neighbour,
  so it gets no contrastive rows.
- The payment application ↔ payment certificate edge is left to the combined-form rule (1c).

### 1c. Combined-form pairs: payment application vs payment certificate (0 rows)
- The field state is read from content only:
  - certificate side: a certified amount (the word "certified" followed within 160 chars by a non-zero amount) AND a certifier
    signature/date (within 160 chars of a certifier cue, a sign/date cue followed within 60 chars by a date value);
  - application side: a certifier cue is present and an amount is present, but there is no certified amount and no certifier signature/date (the certifier block is blank);
  - anything else is indeterminate.
  The cue words come from the v9 definitions and the owner's rule, and a test checks that.
- Only rows whose field state agrees with their teacher label qualify. A pair is one application-side and one certificate-side
  document from the same family (project). Unmatched documents make no pair, because a one-sided set would only push the
  model further toward "application".
- **Qualifying training docs: application side 13, certificate side 0.** Of the 12 train rows labelled payment certificate, 11 are
  indeterminate (no certified amount with a signed or dated certifier block) and 1 reads as application. Of the 37 labelled payment application,
  13 are application side, 22 indeterminate and 1 reads as a signed certificate. **The certificate side is zero. This is the real data gap:
  the training corpus holds no completed and signed certifier block.** So 1c produces 0 pairs, and the payment certificate class
  cannot be fixed from this corpus.

### 1d. Catch-all pairs (183 rows)
- Catch-all labels are correspondence, other, report and register/log (owner task). The instrument labels are the other 18.
- Instrument test: an instrument label's name or one of its unambiguous regional names appears in the first 300 characters of the
  text, the title zone. A catch-all row qualifies when it passes the content gate and fails every instrument test.
- Qualifying train rows: correspondence 65, other 231, report 109, register/log 39. Up to the budget (48) per label goes in:
  48 / 48 / 48 / 39.
- Menu = the row's own catch-all label plus all 18 instrument labels. The other catch-alls are left off, so the loss is spent on
  "named instrument or not".

## Isolation
- Every pair points at a train row with the same doc_id, file_sha256 and label. Pair texts are base train texts, so v7 adds no
  new document text.
- `tests/router/test_v7_pairs.py` runs `router.isolation.require_isolated` (file sha256 / doc id / project) against every sealed file.
  It also checks sha256(text), 12-word shingles and first-300 against the sealed excerpts.
- `data/router/v7/isolation.json` is the box-side check against the full scoring text of all 312 sealed rows. File sha256, text
  sha256, first-300 and project overlaps are all 0. The shared 12-word shingles it lists (boilerplate form or contract text, largest
  share 0.318) come from the base train rows that v6 already trained on. v7 adds none.

## Training
`router.train_router.fit_router_v7(root, examples, val_rows, labels)`, with examples from `router.v7_pairs.v7_training_examples(root)`
(1719 base + 980 pairs = 2699). The recipe is otherwise v6: Qwen/Qwen3.5-4B LoRA r16, alpha 16, lr 4.73e-4, batch 16, seed 0,
max_sequence_length 6656, at most 3 epochs, early stop with patience 1. The epoch is chosen on val macro recall over present classes,
a tie keeps the earlier epoch, and the sealed sets are never read.
