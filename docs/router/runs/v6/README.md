# Router run v6 (class-weighted, Qwen/Qwen3.5-4B LoRA r16, v9 labels)

The committed record of the run. `RUN_RECORD.json` has the summary in one file.

- Run id: `93552b55-3c20-5862-8bbf-2320fb04b236:train:0`
- Chosen checkpoint (epoch 3): `tinker://93552b55-3c20-5862-8bbf-2320fb04b236:train:0/sampler_weights/router-v6-qwen3.5-4b-lora-r16-epoch3`
- Trained on main `3f72300806cf43eb7efcfd1bd9093d0eaccd4dcb` (train 1712 / val 900), config `configs/router/train_v6.json`
- Started 2026-10-07T23:02:30+0400, epoch choice written 2026-10-07T23:28:26+0400

| epoch | val macro recall, present classes (selection) | val accuracy (report only) |
|---|---|---|
| 1 | 0.6622 | 0.7900 |
| 2 | 0.7287 | 0.8733 |
| 3 | 0.7409 | 0.8544 |

Epoch 3 was chosen by `router.balance.select_epoch` on val macro recall before any sealed set was read. Accuracy was never used.

Files: `metrics.json`, `epoch_choice.json` and `checkpoints.json` as written by the run; `RUN_STARTED_train.json`,
`preflight.json`, `preflight.log`, `PRE_RUN_NOTE_train.txt`; `train-v6.log` (the env line keeps only a count of names,
no values); `val_predictions_epoch1-3.jsonl` (doc ids, gold, prediction); `driver/` (the exact training driver).
`epoch_choice.json` still says `sealed_scored: false` because it records the moment of the choice.

Sealed scoring (once, after the choice): `sealed/`. Read `sealed/SEALED_RESULT.md` first. v5 vs v6 per class: `sealed/V5_VS_V6.md`.
No document text is stored here.
