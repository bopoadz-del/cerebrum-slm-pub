# Baselines: held-out shard `governance_fresh_9` (n=9)

This is a separate table for the second held-out shard. `eval/baselines/scores.md` is unchanged, and so are its numbers for the original
shard. Aggregate counts only, with no item text. Shard rules are in `docs/eval/HELD_OUT.md`.

## Shard

- `data/held_out/governance_fresh_9.jsonl`, 9 owner-written governance posture pairs, copied byte for byte. sha256
  `c4b9d9ff4f4cc2c3f6d98c551ab0b9b1b9766d812c28c564003508419d61e830` (seal `governance_fresh_9.jsonl.sha256`, listed in
  `data/held_out/shards.json`, frozen 2026-10-08).
- Labels come from the held-out pair_type rule in `eval/labels.py`. 1 item is `refuse` (REFUSAL_VS_VACUUM), and 8 are `qualify`
  (AUTHORITY_SOURCE, DECISION_LATENCY ×2, PROCEDURAL_VS_TECHNICAL ×2, ESCALATION_CLOSURE ×2, RESOURCE_VARIANCE). No item expects a formula
  id, so a routing pass is an empty route.
- The shard is not in train and not in `pairs.jsonl`. Gate `held-out-shards` found no exact or near duplicate (Jaccard ≥ 0.8) against train,
  the original shard, or the sealed eval sets (regression, temporary works, blind). The highest user-side Jaccard is 0.286, against the
  regression set.

## Scores (scored once, 2026-10-08)

RQ = refusal/qualification passes, Route = formula-routing passes, Leak = fact-leak count.

| Model | Shard | n | RQ | Route | Leak |
|---|---|---|---|---|---|
| Untuned (`base://Qwen/Qwen3-8B`, no system prompt) | governance_fresh_9 | 9 | 0 | 0 | 2 |
| Prompt-only (base + `prompt_only_system_prompt.txt`) | governance_fresh_9 | 9 | 8 | 8 | 0 |
| Run 4 epoch 2 tuned (reported) | governance_fresh_9 | 9 | 0 | 0 | 1 |
| Run 4 epoch 2 tuned + prompt (served artifact) | governance_fresh_9 | 9 | 8 | 9 | 0 |

Per item (`governance_fresh_9/*.items.json`, grades only):
- The one `refuse` item (line 2) fails RQ in both prompted cells.
- Prompt-only misses Route on one `qualify` item (line 4).
- The untuned leaks are one currency figure and one site name. The tuned-without-prompt leak is one site name.

Original shard, quoted from the existing records and not re-scored. Protocol v2, `held_out` n=69:

| Model | RQ | Route | Leak | Source |
|---|---|---|---|---|
| Untuned | 0 | 0 | 17 | `eval/baselines/scores.md` (Protocol v2) |
| Prompt-only | 51 | 61 | 0 | `eval/baselines/scores.md` (Protocol v2) |
| Run 4 epoch 2 tuned (reported) | 0 | 0 | 23 | `docs/runs/run4-cap3-weight-x3-stop-gate.md` |
| Run 4 epoch 2 tuned + prompt (served) | 62 | 68 | 0 | `docs/runs/run4-cap3-weight-x3-stop-gate.md` |

The stop-gate (`eval/STOP_GATE.md`) still reads only the original `held_out` and `blind` sets. This shard is reported and does not gate.

## How these numbers were produced

- Command: `scripts/eval/run_baseline_eval.py` (untuned, prompt-only) and `scripts/eval/run_tuned_eval.py --checkpoint …` (tuned cells), each
  with `--shard governance_fresh_9`. Only this shard was sampled, and the original shard and the blind set were not re-sampled.
- Model `Qwen/Qwen3-8B`. Sampling config `configs/model/qwen3-8b.json`: temperature 0.7, top_p 0.8, top_k 20, max_tokens 1024, seed 0,
  thinking off. These are the same settings as the recorded baselines.
- Protocol v2, as in the run 4 record. Refusal/qualification and fact-leak are sampled context-free. Routing is sampled with
  `data/retrieval/formula_registry.json` attached and constrained formula ids. No Fork context.
- Graders: the committed refusal/qualification, fact-leak and formula-routing graders, unchanged.
- Adapter: run 4 epoch 2, PASS recorded in PR #22: `tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch2`
  (Tinker run `d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0`).
- Code: main `729a358` plus this PR's shard harness. File sha256s, per-cell start/end times (UTC+4) and raw-reply sha256s are in
  `governance_fresh_9/run.json`. Cells ran 2026-10-08 20:34:50–20:39:21 +04.
- Raw replies stay on the box outside the repo, as for every recorded eval. The repo keeps the aggregate scores (`*.scores.json`) and the
  per-item grades (`*.items.json`), which hold no item text and no reply text.
- An earlier start at 20:20:40 +04 hung on its first sample because of the Tinker client transport. It returned no sample and wrote
  nothing. The rerun with the httpx transport, the same workaround the earlier run drivers use, is the only scoring.
