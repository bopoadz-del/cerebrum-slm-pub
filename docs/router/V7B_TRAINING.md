# Router v7b: repair attempt 1 of 2

v7 was scored once on the 312 sealed rows and was not promoted (see `docs/router/runs/v7/README.md`). v7b changes three generic
rules in the pair generator. Nothing else changes: the v6/v7 training recipe, the tempered sampler rule (exponent 0.5, cap 3x natural
share, weights in a data file), the full 22-label prompt, the parser, the gate and the sealed sets.

## What changes, and why (from the v7 confusion matrix and the v7 val confusion)

| v7 failure | generic cause | v7b rule (configs/router/pairs_v7b.json) |
|---|---|---|
| correspondence -> other / report, other and NCR -> report, other precision 0.417 | 1d set each catch-all against the instrument labels only, so catch-all vs catch-all was never practised | `catch_all.sibling_pairs`: each selected catch-all row also gets a two-label menu against one sibling catch-all (least-used sibling) |
| register/log -> drawing, payment application over-predicted | 1b spread a class budget round-robin over all Is-not neighbours, so the neighbour a document's own text names got no more practice than any other | `contrastive.neighbour_choice = named_in_head_first`: rows whose first 3000 characters name an Is-not neighbour fill the budget first and get the least-used named neighbour, capped at 3x the round-robin share per edge; other rows keep the least-used neighbour |
| correspondence collapse also on val (recall 0.10) | 52 of 80 training correspondence rows come from one project; hash order let one project fill a class budget | `row_order = project_round_robin`: rows are interleaved across projects before the budget is taken |

Payment certificate (recall 0.40, 9 of its rows predicted as payment application) is a data gap. The train split has no
certificate-side rows that the combined-form rule can pair, so v7b cannot meet the payment certificate criterion. It stays unreachable
until certified, signed payment certificates enter train.

## Data

- `data/router/v7b/pairs_contrastive.jsonl`: 797 rows (329 paired with a neighbour named in their own text)
- `data/router/v7b/pairs_combined_form.jsonl`: 0 rows (same rule as v7; still no certificate-side rows)
- `data/router/v7b/pairs_catch_all.jsonl`: 183 rows (same rule as v7)
- `data/router/v7b/pairs_catch_all_sibling.jsonl`: 183 rows
- `data/router/v7b/class_weights.json`: 2882 examples, 22 classes, weights recomputed by the v7 rule from these counts
- Every pair row is a base training row (same text) with a restricted menu, so the v7 isolation report (`data/router/v7/isolation.json`) covers v7b.
  `tests/router/test_v7b_pairs.py` asserts that every v7b example text is a base train text.

Regenerate or check: `PYTHONPATH=scripts python -m router.v7_pairs --root . --train-config configs/router/train_v7b.json [--check]`.
v7 still regenerates byte for byte (`--check` without `--train-config`).

## Run plan

Train once on the merged tree with `configs/router/train_v7b.json` (checkpoint `router-v7b-qwen3.5-4b-lora-r16`). Pick the epoch on val macro recall
over present classes. Score once on the combined 312 sealed rows plus the name-blind check. The run record goes under `docs/router/runs/v7b/`
and is written after scoring. If v7b misses, at most one more repair attempt (v7c) follows, then park.
