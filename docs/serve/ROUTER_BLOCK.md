# Router block

The router block classifies a document from its first 512 content tokens into the label set in `configs/router/labels.json`. The user message is that text. Filename, path, document id, project, and `fork_declared_type` are not inputs.

The served adapter is a Qwen3.5-4B LoRA, rank 16, alpha 16, two epochs, 168 steps. Tinker does not serve `Qwen/Qwen3-4B`, so the run used `Qwen/Qwen3.5-4B`. Max sequence length for that run is 3072. The serve file is `configs/serve/router-epoch2.json`.

Status: `trained`. sealed gate: FAIL (gate defect: condition (b) unsatisfiable as written); router NOT marked passed. Sealed gate version 2 applies only to a future fresh sealed set. The v5 result is unchanged.

Checkpoint: `tinker://b0857111-57c4-56f6-bf9b-ed3f8021826b:train:0/sampler_weights/router-qwen3.5-4b-lora-r16-epoch2`

Epoch 2 val accuracy is 0.906. The majority-class gate passes because the model is not a constant predictor. The constant `drawing` baseline fails that gate. Per-label recall is in `docs/router/DATASET_CARD.md`.

The sealed set `data/eval/router_unseen/sealed_labels.jsonl` was scored once (macro recall 0.715, accuracy 0.672, 43/64). `sealed_scored` is true. `sealed_scoring` stays false, so the set is not scored again and not tuned on. The one-time record is `docs/router/runs/v2/sealed/`. VO/CO recall is 0/5. 3 outputs said 'change order', which the val parser maps to unsure, and they were not remapped. Those rows were not re-parsed after the alias parser landed.

Human agreement is done (0.875, kappa 0.864). Training proceeded before that check by owner order. The majority-class gate and the agreement threshold are unchanged.
