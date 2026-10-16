# Protocol v2 re-sample (routing with attached formula registry)

Aggregate counts only. No eval item text.

## Protocol change (PR #19, 1bfe089)

- Routing is scored only with an attached formula registry (held-out: data/retrieval/formula_registry.json; blind: HDS-5 equations from eval/graders/data/hds5_equations.json).
- Constrained decoding on formula ids from the attached context is ON for routing samples.
- Refusal/qualification and fact-leak stay context-free (dual sample per item).
- Stop-gate compares **tuned + system prompt** only against the new prompt-only lines; tuned-without-prompt is reported, not gated.
- Equation names: prose from section heading / Where-block / intro (not garbled PDF math). Extractor: scripts/eval/extract_hds5_equations.py. Registry: **18** equations, PDF pages **[91, 92, 93, 94, 96, 97, 190, 191, 192]**.

## Checkpoints / models

- Base / prompt-only: ase://Qwen/Qwen3-8B
- Run-3 epoch 1: 	inker://b5a300e4-7c08-5a19-809e-4b006deee3a5:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch1
- Sampling via shared-client runner (httpx forced, per-sample progress). 258 samples = 129 items × 2.

## Scores (new protocol)

Held-out n=69, blind n=60. RQ = refusal/qualification passes, Route = formula-routing passes, Leak = fact-leak count.

| Model | Held-out RQ | Held-out Route | Held-out Leak | Blind RQ | Blind Route | Blind Leak |
|---|---|---|---|---|---|---|
| Base untuned | 0 | 0 | 17 | 2 | 0 | 25 |
| Prompt-only | 51 | 61 | 0 | 3 | 17 | 0 |
| Run-3 ep1 tuned (reported) | 0 | 0 | 18 | 6 | 0 | 30 |
| Run-3 ep1 tuned + prompt (gated) | 62 | 66 | 0 | 7 | 17 | 0 |

## STOP GATE (eval/STOP_GATE.md)

Bar = new prompt-only (held Route 61, blind Route 17; RQ 51 / 3; leak 0).

| Checkpoint | leak held | leak blind | RQ held | RQ blind | Route held | Route blind | Result |
|---|---|---|---|---|---|---|---|
| Run-3 ep1 tuned + prompt | ok | ok | ok (62>51) | ok (7>3) | ok (66>61) | fail (17≯17) | FAIL |

Decision: **FAIL** — blind routing ties prompt-only (tie does not beat). No serving from run-3 ep1 under the new protocol. Proceeded to run 4 (PR #20).

## Artifacts

JSON under operator slm-runs/protocol-v2/: ase.json, prompt.json, 	uned_ep1.json, 	uned_ep1_prompt.json.
