# Model card — Qwen3-8B instruct (base)

This card records the base model for the construction posture SLM. The model refuses, qualifies, names the missing input, and routes to a formula. Figures, clauses, brands, site facts, and formula values are not stored in the weights. Retrieval is The Fork. Formula values come from a formula executable.

## Identity

| Field | Value |
| --- | --- |
| Model id | `Qwen/Qwen3-8B` |
| Role here | Instruct base |
| Upstream base | `Qwen/Qwen3-8B-Base` (this checkpoint is the post-trained model, not that base) |
| Tinker id | `Qwen/Qwen3-8B` |
| Type | Causal language model |
| Training stage | Pretraining and post-training |
| Parameters | 8.2B (6.95B non-embedding) |
| Layers | 36 |
| Attention | GQA, 32 query heads, 8 key/value heads |
| Context | 32,768 tokens native |
| Thinking | Hybrid. Sampling for this project turns thinking off so the reply is the answer. |
| Publisher card | https://huggingface.co/Qwen/Qwen3-8B |
| Tinker model list | https://tinker-docs.thinkingmachines.ai/tinker/models/ |

`Qwen/Qwen3-8B-Base` is a different checkpoint. Sampling and any later training must use `Qwen/Qwen3-8B`.

## Licence

Confirmed from the model repository `LICENSE` file on 4 October 2026.

- The file is the Apache License, Version 2.0, with the appendix notice `Copyright 2024 Alibaba Cloud`.
- Source: https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE
- Saved text: `configs/model/LICENCE-Qwen3-8B.txt`
- SHA-256: `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`

The saved text is the licence. It has no non-commercial restriction. Commercial use is permitted under those terms. Machine-readable fields live in `configs/model/qwen3-8b.json`.

## Sampling

The client is `src/model/sampling.py`. It reads `TINKER_API_KEY` from the environment and passes `Qwen/Qwen3-8B` to Tinker `create_sampling_client`. The key is not written into this repo.

Non-thinking settings, from the publisher's non-thinking recommendation, are fixed in `configs/model/qwen3-8b.json`: temperature 0.7, top_p 0.8, top_k 20, plus seed 0 and max_tokens 1024. `enable_thinking` is false.

On 4 October 2026 the environment that prepared this change had no `TINKER_API_KEY`. The sampling call was not made. `sample_messages` raises `TinkerKeyMissing` until the key is present in the environment. That does not block the licence, the graders, or the prompt-only system prompt.

## What this base is for

Posture only: state the basis (`as-designed`, `as-built`, `as-operating`, or `unspecified`), name a missing input, refuse an unsourced figure, and route to a formula id with its required inputs. No training run is part of this card.

## Served artifact

The construction posture adapter is served with the committed system prompt in
`eval/baselines/prompt_only_system_prompt.txt`. The stop-gate evaluates that
paired artifact (tuned weights + system prompt). A tuned checkpoint without the
system prompt is reported for diagnosis and is not a serve decision.

Pinned serve path (run 4 epoch 2, protocol v2 PASS):

`tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch2`

Config and client: `configs/serve/posture-epoch2.json`, `src/posture_model/`.
Blind refusal/qualification: **7/60** (`KNOWN_LIMITATIONS.md`).
