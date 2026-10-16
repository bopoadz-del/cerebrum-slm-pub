# Router v6 training mechanism (pr11)

Status: mechanism only. v6 is not trained until the new train rows (another workstream) merge.

## Why
Run v5 never predicted site instruction, VO/CO or claim on val (val n 8, 1 and 1). Its train rows are imbalanced: drawing 400 and other 300, against VO/CO 5 and claim 10. Training on unweighted cross-entropy lets the large classes take most of the gradient. That is a training defect, not a data rule. The fix is one generic mechanism computed from label counts.

## Choice: class-weighted loss, not class-balanced sampling
Tinker's `cross_entropy` loss is `sum(-logprob * weights)` over target tokens. The router already passes per-token `weights` (0 on the prompt, 1 on the label tokens). Class weighting multiplies the label-token weights of each row by its class weight. It changes nothing else, so this is the cleaner of the two options:
- Every row is seen once per epoch. No row is duplicated, and no row is chosen for oversampling. Balanced sampling with replacement would repeat the 5 VO/CO rows many times per epoch and over-fit them.
- Epoch length, step count, batch size and learning rate stay as in v5. The weights are renormalised so the mean over training rows is 1.0, so the gradient mass per epoch matches an unweighted run.
- It is deterministic for a fixed seed and needs no new sampler state.

## Rule (`scripts/router/balance.py`, config `configs/router/train_v6.json`)
- N = training rows, K = classes present in training, n_c = rows of class c.
- raw_c = N / (K · n_c). This is 1.0 for every class when classes are balanced.
- capped_c = min(raw_c, cap) with cap = 5.0. A rare class gets at most 5× the balanced weight, so a 5-row class cannot dominate.
- w_c = capped_c / (Σ n_c · capped_c / N). The mean weight over training rows is 1.0.
- Nothing names a label, project, document or row. Labels are only keys in a count. Tests check that renaming classes leaves the weights unchanged and that no label string appears in the mechanism code or config.

Example on the current train split (1602 rows), for illustration only: drawing 0.198, other 0.264, report 0.644, site instruction 1.650, claim 5.438 (capped), VO/CO 5.438 (capped).

## Selection and early stop
- Selection metric: validation macro recall over the classes present in val (`val_macro_recall_present_classes`, `router.balance.macro_recall_present`). It is never accuracy, and the config loader refuses any other metric.
- The epoch is chosen by that metric, and a tie keeps the earlier epoch. This happens before any sealed score.
- Early stop: after each epoch the saved sampler labels val. Training stops when an epoch does not beat the best earlier epoch (patience 1), or at 3 epochs. `epochs` must be 1..3.
- max_sequence_length stays 6656. The recipe is otherwise v5's: Qwen/Qwen3.5-4B LoRA r16, alpha 16, lr 4.73e-4, batch 16, seed 0.

## Entry point
`router.train_router.fit_router_v6(root, train_rows, val_rows, labels)`. It returns the class weights, the per-epoch history, the selected epoch and its sampler. `fit_router` and `_fit` keep the v5 behaviour when `class_weight` and `on_epoch` are not passed.
