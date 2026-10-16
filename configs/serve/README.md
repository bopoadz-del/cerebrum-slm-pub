# Serve configs

Pinned serve artifacts for the construction posture model.

| Config | Checkpoint | Stop-gate |
| --- | --- | --- |
| `posture-epoch2.json` | `tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch2` | PASS (run 4, protocol v2) |

The system prompt at `eval/baselines/prompt_only_system_prompt.txt` is part of the served artifact. Load and sample through `src/posture_model/` (see `docs/serve/LOAD_AND_SAMPLE.md`).
