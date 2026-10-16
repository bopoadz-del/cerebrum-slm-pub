# Run 3 (rank 16, phase-1 data): per-epoch scores and STOP GATE 5.4

Aggregate counts only. No eval item text.

## Run

- Config `configs/lora/qwen3-8b-posture-r16.json` at main `e658ecb` (#17): `Qwen/Qwen3-8B`, rank 16, LR 0.000473, 2 epochs, batch 16, 266 steps, 2,122 train pairs (2,122 with the eval system prompt), `posture_json_v1` targets.
- Tinker run `b5a300e4-7c08-5a19-809e-4b006deee3a5:train:0`.
- Epoch samplers: `tinker://b5a300e4-7c08-5a19-809e-4b006deee3a5:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch1` and `...-epoch2`. Final state: `.../weights/posture-qwen3-8b-lora-r16`.
- Each epoch scored with `scripts/eval/run_tuned_eval.py` at `e658ecb` (current graders and labels), no system prompt and with `--prompt-only-system-prompt`, no Fork context. Gate run with `scripts/eval/run_stop_gate.py` against `eval/baselines/scores.md`.

## Scores

Held-out n=69, blind n=60. RQ = refusal/qualification passes, Route = formula-routing passes, Leak = fact-leak count.

| Model | Held-out RQ | Held-out Route | Held-out Leak | Blind RQ | Blind Route | Blind Leak |
|---|---|---|---|---|---|---|
| Base untuned (scores.md) | 0 | 0 | 14 | 3 | 0 | 26 |
| Prompt-only (scores.md) | 51 | 46 | 0 | 3 | 19 | 0 |
| Epoch 1 tuned | 0 | 0 | 17 | 6 | 0 | 30 |
| Epoch 1 tuned + prompt | 62 | 64 | 0 | 7 | 15 | 0 |
| Epoch 2 tuned | 0 | 0 | 7 | 4 | 0 | 29 |
| Epoch 2 tuned + prompt | 61 | 66 | 0 | 0 | 18 | 0 |

## STOP GATE 5.4 (`eval/STOP_GATE.md`)

Checks: 1 held-out leak 0; 2 blind leak 0; 3 held-out RQ > 51; 4 blind RQ > 3; 5 held-out Route > 46; 6 blind Route > 19.

| Checkpoint | 1 | 2 | 3 | 4 | 5 | 6 | Result |
|---|---|---|---|---|---|---|---|
| Epoch 1 tuned | fail (17) | fail (30) | fail (0) | ok (6) | fail (0) | fail (0) | FAIL |
| Epoch 1 tuned + prompt | ok | ok | ok (62) | ok (7) | ok (64) | fail (15) | FAIL |
| Epoch 2 tuned | fail (7) | fail (29) | fail (0) | ok (4) | fail (0) | fail (0) | FAIL |
| Epoch 2 tuned + prompt | ok | ok | ok (61) | fail (0) | ok (66) | fail (18) | FAIL |

Decision: FAIL on every epoch in both modes. No checkpoint from run 3 is served. Blind formula routing stays at or below prompt-only (15 and 18 vs 19). Under the early-stop rule in `configs/lora/README.md`, the no-prompt epoch 1 already has held-out leak above 0. This was the last training run allowed.