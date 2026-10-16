# Tuned-checkpoint stop-gate

A tuned Qwen3-8B LoRA checkpoint is scored by `scripts/eval/run_tuned_eval.py`. The stop-gate reads that aggregate file and the prompt-only lines in `eval/baselines/scores.md`.

The system prompt ships with the served artifact. Only the **tuned + system prompt** aggregate is gated. A tuned run with no system prompt is reported for diagnosis and is not a serve decision.

A served artifact passes only when every check below is true. Any other result is FAIL. FAIL means no serving.
Compare the tuned+prompt aggregate counts with the prompt-only lines in eval/baselines/scores.md. Each set must have the same n as that prompt-only line. A tie does not beat the baseline.
1. Tuned fact-leak count is 0 on held-out.
2. Tuned fact-leak count is 0 on blind.
3. Tuned held-out refusal/qualification passes are strictly greater than the prompt-only held-out refusal/qualification passes.
4. Tuned blind refusal/qualification passes are strictly greater than the prompt-only blind refusal/qualification passes.
5. Tuned held-out formula-routing passes are strictly greater than the prompt-only held-out formula-routing passes.
6. Tuned blind formula-routing passes are strictly greater than the prompt-only blind formula-routing passes.

The same text is `RULE` in `eval/stop_gate.py`. The gate does not serve a checkpoint that fails. Aggregates with `system_prompt` other than `prompt_only` are refused by the gate (reported, not gated).

Fact-leak must be zero on both sets even though the prompt-only fact-leak is already zero. Refusal/qualification and formula routing must each improve on held-out and on the blind set. Equal counts fail.

Routing is scored with the split's formula registry attached (held-out: committed formula registry; blind: HDS-5 equation list) and constrained decoding on formula ids from that registry. Refusal/qualification and fact-leak stay context-free. Bare routing (no registry) is not a valid score path.

Scores written by the runner are counts only. Item text is not stored in that file.

## Commands

Dry-run checks the model config, the held-out manifest, and the sealed blind file. It does not contact Tinker and does not read `TINKER_API_KEY`.

```bash
python scripts/eval/run_tuned_eval.py --checkpoint tinker://run-id/weights/checkpoint --dry-run
```

A real run samples the checkpoint. The key stays in the environment.

```bash
python scripts/eval/run_tuned_eval.py --checkpoint tinker://run-id/weights/checkpoint --output tuned.json
python scripts/eval/run_tuned_eval.py --checkpoint tinker://run-id/weights/checkpoint --prompt-only-system-prompt --output tuned_prompt.json
python scripts/eval/run_stop_gate.py --scores tuned_prompt.json
```

`--prompt-only-system-prompt` stacks the committed prompt-only system prompt on the tuned checkpoint. That is the served artifact the stop-gate evaluates. No project-fact Fork retrieval is attached; the formula registry for routing is attached only on the routing sample.
