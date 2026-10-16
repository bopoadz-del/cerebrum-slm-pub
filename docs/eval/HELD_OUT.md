# Held-out shards (posture model)

This page covers the held-out pairs of the posture model (the Construction SLM adapter). It does not cover the document router.

## Rule

1. **Held-out shards are frozen.** A listed shard is never edited, re-ordered, appended to or re-split. Its sha256 and row count are recorded in
   `data/held_out/shards.json`, and the gate fails if either changes.
2. **New owner pairs go to a new shard.** New pairs are copied byte for byte into a new file under `data/held_out/`. They get a `.sha256` seal
   beside the file (sha256sum style) and a new entry in `shards.json`. They are never added to an existing shard and never to train.
3. **Shards merge only on a baseline reset.** Merging two shards changes the set every recorded number was measured on. It is allowed only
   together with a baseline reset, which means re-sampling every recorded cell (untuned, prompt-only, tuned, tuned + prompt) on the merged
   set and recording the reset in `eval/baselines/`. Without a reset, each shard keeps its own line and its own baselines.

No held-out pair from any shard may be in train, whether as the same pair, the same user turn, or an augmentation child of a held-out
parent.

## `data/held_out/shards.json`

The shard list is data. No shard name, file name or pair text appears in code. Schema `held_out_shards/v1`, with one object per shard:

| field | meaning |
|---|---|
| `name` | shard id, used as the line label in eval output |
| `path` | the shard file, under `data/held_out/` |
| `sha256` | sha256 of the file bytes at freeze |
| `count` | number of non-blank rows |
| `frozen` | must be `true` |
| `created` | date the shard was frozen (YYYY-MM-DD) |
| `original` | `true` for exactly one shard: the phase-1 split `data/held_out/pairs.jsonl`, whose numbers are in `eval/baselines/scores.md` |
| `record_schema` | `system_user_assistant` or `instruction_input_output`; the reader maps either onto the chat schema |
| `routing_registry` | which formula registry routed items attach (`held_out` = `data/retrieval/formula_registry.json`) |
| `near_dup_at_freeze` | near-duplicate count measured at freeze (see below); the gate fails if it grows |
| `baseline_record` | the file that records this shard's baselines |
| `provenance` | where the rows came from |

## Shards

| shard | file | rows | created | baselines |
|---|---|---|---|---|
| `held_out` (original) | `data/held_out/pairs.jsonl` | 69 | 2026-10-05 | `eval/baselines/scores.md`, `docs/runs/run4-cap3-weight-x3-stop-gate.md` |
| `governance_fresh_9` | `data/held_out/governance_fresh_9.jsonl` | 9 | 2026-10-08 | `eval/baselines/governance_fresh_9.md` |

## Eval harness

- `eval/shards.py` reads `shards.json`, checks each frozen shard against its sha256 and count, and labels rows with the held-out pair_type
  rule (`eval/labels.py`). The original shard's labels are exactly `label_eval_sets()`'s `held_out` labels.
- `scripts/eval/run_tuned_eval.py` and `scripts/eval/run_baseline_eval.py`, default run: the stop-gate sets (`held_out`, `blind`) are sampled
  and written exactly as before. The stop-gate payload schema is unchanged. Then every added shard is sampled with the same system prompt,
  sampling config, graders and dual-sample routing protocol. All shards are written to `<output stem>.shards.json` and printed one line per
  shard. The original shard's line is the `held_out` tally from the same run, not a second sample.
- `--shard NAME` (repeatable) samples only the named shards and does not touch the stop-gate sets. `--item-results` writes per-item grades with
  no item or reply text. `--replies` writes the raw replies to a path outside the repo.
- The stop-gate (`eval/STOP_GATE.md`) still reads only `held_out` and `blind`. Added shards are reported, not gated.

## Gates

`scripts/gates/held_out_shards.py` runs in `python -m gates.run` (CI). For every listed shard it checks:

- the file exists, matches its sha256 and count, and an added shard's `.sha256` seal matches;
- every `.jsonl` under `data/held_out/` is listed, and the original shard is the split the eval labels read;
- no shard row shares a content hash or a user turn (case and whitespace folded) with any row in a train file, meaning every `train_file`
  named in `configs/lora/*.json` plus everything under `data/train/`; and no row is in two shards;
- near-duplicates: user-side Jaccard over lower-cased word tokens ≥ 0.8 (the augmentation gate's `OVERLAP_LIMIT`) against train rows, other
  shards and sealed eval items. The count may not exceed `near_dup_at_freeze`. The original shard was frozen with 3. These are templated
  formula questions that differ only in the subject word, with sibling templates in train, and they predate this rule. Every added shard
  is frozen with 0.

`scripts/gates/augment.py` treats every listed shard as held-out. A train child whose parent is a row of any shard fails, and a shard row
that carries augmentation fails.
