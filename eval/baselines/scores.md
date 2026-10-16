# Baseline scores

Untuned, held-out, n=69: refusal/qualification 0, fact-leak rate 14/69, formula-routing accuracy 0.
Untuned, blind, n=60: refusal/qualification 3/60, fact-leak rate 26/60, formula-routing accuracy 0.
Prompt-only, held-out, n=69: refusal/qualification 51/69, fact-leak rate 0, formula-routing accuracy 46/69.
Prompt-only, blind, n=60: refusal/qualification 3/60, fact-leak rate 0, formula-routing accuracy 19/60.

## Held-out re-sample (phase-1 data)

The phase-1 data build added 4 confirmed temporary-works items to held-out (65 -> 69). The two held-out lines above were sampled again on all 69 items with the corrected grader, the same prompt-only system prompt, the same sampling config, and no Fork context. Earlier n=65 lines: untuned 0, 18/65, 0; prompt-only 47/65, 0, 42/65. Those 69 held-out replies were graded with the current grader and labels (symbol aliases, registry required inputs for routed items). A fresh blind sample taken at the same time and graded with the current grader scored prompt-only 3/60, 0, 18/60 and untuned 2/60, 24/60, 0. The prompt-only blind line keeps the higher bar of the two samples on each count: refusal/qualification 3/60 (was 2/60), formula-routing 19/60. The untuned blind line is unchanged.

## Grader correction (phase-1 return)

The refusal/qualification grader was corrected by shape rules (block classes across the two vocabularies, both name-list labels, separator folding). See `docs/audit/phase1-grader-and-format-audit.md`. The refusal/qualification counts above were re-scored with the corrected grader on a fresh sample at commit `711021c` (same prompt, same sampling config, no Fork context). Held-out refusal/qualification did not change (47 and 0). Blind refusal/qualification moved 0 -> 2 for prompt-only and 2 -> 3 for untuned. The fact-leak and formula-routing graders are unchanged, so those recorded counts stand.

## Grader and label corrections (owner order 11:29)

Two more shape corrections, applied identically to every model, were re-scored on the same stored run-1 replies (no resampling): base, prompt-only, tuned, and tuned with the prompt.

1. Symbols vs names. The refusal/qualification grader also accepts a descriptive name for a named input when it identifies one symbol in the blind corpus's own LIST OF SYMBOLS (`eval/graders/data/hds5_symbols.json`, 146 entries extracted by `scripts/eval/extract_symbol_table.py` from FHWA-HIF-12-026, pages 19-23). No stored count changed: the descriptive names the models wrote were either ambiguous in the table (for example a bare "area" or "velocity" phrase) or named a different symbol.
2. Held-out routing labels. The 17 routed held-out items now expect the required inputs of their formula in `data/retrieval/formula_registry.json` (`eval/registry.py`, the same rule as the training targets) instead of an empty set. No stored count changed: no stored reply routed any of those 17 items to the right id before or after, so prompt-only held-out routing stays 42/65.

The recorded lines above therefore stand.

## How these numbers were produced

Model `Qwen/Qwen3-8B`. Repo commit sampled: `a0cd715c3d7999b2c20cc2383aa4e1e2c9363b3a`. Sampling used `sample_messages`. The prompt-only run used the existing `eval/baselines/prompt_only_system_prompt.txt`. Grading used the existing refusal, fact-leak, and formula-routing graders. The functions `run_untuned_baseline` and `run_prompt_only_baseline` still raise `BaselineNotRun`, so they were not what produced these numbers. No Fork context was attached.

Held-out items have no block field. Refusal labels were taken from `pair_type`: refuse for `REFUSAL_SCOPE`, `REFUSAL_VS_VACUUM`, `FORCING_THE_NUMBER`, `bare_rate_not_answer`, and `why_formula_cannot_answer`; block none for `which_formula_applies`; otherwise qualify. A formula id was expected only for `which_formula_applies`, `what_must_be_known_first`, and `why_formula_cannot_answer`. Every other held-out item expected an empty route.

Blind labels used each item's shape: `expected_block` and `named_inputs` for refusal; `formula_id` (null means empty) and `named_inputs` as required inputs when an id is set, otherwise an empty route.

## Label method

`eval/labels.py` is that method in code. `run_untuned_baseline` and `run_prompt_only_baseline` call it and still raise `BaselineNotRun`. They do not sample again, and the counts above are unchanged. A tuned run uses the same method. Held-out required inputs of a routed item come from the committed formula registry, because the gold has no required-input list. The stop-gate that compares a tuned run with the prompt-only lines is `eval/STOP_GATE.md`.

## Protocol v2 (routing with attached formula registry)

Canonical stop-gate bar after PR #19. Routing sampled with an attached formula registry and constrained formula ids; refusal/qualification and fact-leak stay context-free. Re-sampled on main `1bfe089` / runner shared-client.

Untuned, held-out, n=69: refusal/qualification 0/69, fact-leak rate 17/69, formula-routing accuracy 0/69.
Untuned, blind, n=60: refusal/qualification 2/60, fact-leak rate 25/60, formula-routing accuracy 0/60.
Prompt-only, held-out, n=69: refusal/qualification 51/69, fact-leak rate 0, formula-routing accuracy 61/69.
Prompt-only, blind, n=60: refusal/qualification 3/60, fact-leak rate 0, formula-routing accuracy 17/60.

See `docs/runs/protocol-v2-resample-run3-ep1.md` for the six-cell table including run-3 ep1 (gate FAIL: blind route 17=17).
