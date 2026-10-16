# Load and sample the served posture artifact

## Pinned checkpoint

```
tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch2
```

Config: `configs/serve/posture-epoch2.json`. System prompt: `eval/baselines/prompt_only_system_prompt.txt` (part of the artifact). Stop-gate: PASS under protocol v2 (`docs/runs/run4-cap3-weight-x3-stop-gate.md`).

## Python

```python
import os
from posture_model import load_serve_artifact, sample_posture

# TINKER_API_KEY must be in the environment. It is never read from the repo.
artifact = load_serve_artifact()
assert artifact.checkpoint.endswith("posture-qwen3-8b-lora-r16-epoch2")

# Fork must attach context. Empty string means retrieval returned no hits.
reply = sample_posture(
    "What is the concrete strength class for this pour?",
    fork_context="",  # required kwarg
)

# When a formula registry is attached, constrain formula ids:
reply = sample_posture(
    "Route the cantilever UDL deflection check.",
    fork_context=registry_text,  # from The Fork / formula registry
    allowed_formula_ids=registry_ids,
)
```

`sample_posture` raises `ForkContextRequired` if `fork_context` is `None`. It raises `TinkerKeyMissing` if `TINKER_API_KEY` is absent.

## Constrained decoding

When `allowed_formula_ids` is passed, a `formula_id` outside that set is cleared before the reply is returned (`src/decoding/constrained.py` via `constrain_posture_response`). Attach the same registry The Fork would use for routing. Citation ids outside an attached context must not be emitted by the Fork pipeline (`src/retrieval/pipeline.py`).

## What this block does not do

- It does not retrieve. Call The Fork first (`src/retrieval/fork.py` documents the retrieve / formula entry points).
- It does not run formulas. Fork's formula executable returns values.
- It does not read `data/eval/unseen`.

## HTTP fallback

When the native Tinker SDK cannot load (for example Windows Application Control blocking pyqwest), sample_posture falls back to Tinker's OpenAI-compatible chat endpoint (posture_model.http_sample.sample_messages_http). The same checkpoint path and TINKER_API_KEY are used. Thinking is requested off (
easoning_effort=none).
