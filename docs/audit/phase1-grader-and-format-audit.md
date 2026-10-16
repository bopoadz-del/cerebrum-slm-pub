# Phase-1 return after STOP GATE 5.4: grader and training-format audit

Shape-level only. No blind question text and no blind answer text is in this file.
Blind items are referred to by their 1-based line in the sealed file and their gold shape.

## 1. Blind refusal/qualification grader audit

Run 1 scored prompt-only 0/60 and tuned+prompt 0/60 on blind refusal/qualification.
Ten blind items with a refusal-type shape were read side by side with the replies of
prompt-only and tuned+prompt (sampled at main `711021c`, committed prompt, committed sampling config).

| # | Blind line | Gold block | Gold names (kind) | Prompt-only reply (shape) | Tuned+prompt reply (shape) | Miss class |
|---|---|---|---|---|---|---|
| 1 | 1 | false_premise | 3 symbols (binding) | block=qualify; symbols listed with descriptions | block=qualify; formula id invented | a: model did not refuse the premise |
| 2 | 2 | false_premise | 3 symbols (binding) | block=qualify; symbols in both lists | block=qualify; symbols in both lists | a: model did not refuse the premise |
| 3 | 3 | false_premise | 2 symbols (binding) | block=qualify; symbols under required_inputs | block=qualify; symbols under missing_inputs | a: model did not refuse the premise |
| 4 | 11 | refusal | 3 symbols (unsupported object) | block=qualify; one descriptive name | block=qualify; one descriptive name | a: model qualified instead of refusing |
| 5 | 12 | refusal | 1 snake_case name | block=qualify; descriptive name | block=qualify; unrelated name | a: model qualified instead of refusing |
| 6 | 31 | missing_input | 1 symbol | block=qualify; descriptive name, not the symbol | block=qualify; different descriptive name | b on block (qualify vs missing_input), a on names |
| 7 | 32 | missing_input | 1 symbol | block=qualify; descriptive name, not the symbol | block=qualify; descriptive name | b on block, a on names |
| 8 | 33 | missing_input | 1 symbol | block=qualify; descriptive name, not the symbol | block=qualify; descriptive name | b on block, a on names |
| 9 | 21 | qualifier | 1 snake_case switch | block=qualify; switch named with spaces/hyphen | block=qualify; switch named with hyphens | b: block word and separator spelling only |
| 10 | 22 | qualifier | 1 snake_case switch | block=qualify; empty missing_inputs | block=qualify; two split names | b on block, a on names |

Class a: the model was wrong under the reply contract. Class b: the grader rejected a reply
that met the contract, because of the shape of the words, not their content.

### Grader shape mismatches found (class b)

1. **Block vocabulary.** The blind gold uses `false_premise`, `refusal`, `missing_input`, `qualifier`, `formula`.
   The reply contract (the committed prompt-only system prompt) only allows `refuse`, `qualify`, `none`.
   The grader compared the two strings exactly, so no contract-following reply could ever pass a blind item.
   This is why prompt-only scored 0/60.
2. **Name-list label.** The grader looked for named inputs only under `missing_inputs`. The contract also has
   `required_inputs`, and for a false-premise or formula item the binding naturally sits there.
3. **Separator spelling.** `exit-loss method` and `exit_loss_method` were different names.

### Fix (shape rules, no per-item logic): `eval/graders/refusal.py`

- Blocks compare by class: refuse = {refuse, refusal, false_premise}; qualify = {qualify, qualifier, missing_input};
  none = {none, formula}. Classes never merge. An unknown word compares by its own spelling.
- A prose reply with no block field passes on the gold word or its class word, as a whole word.
- Named inputs are read from every name-list field (`missing_inputs` and `required_inputs`).
- Names compare after case folding and after runs of spaces, hyphens, and underscores collapse.
  Spelling is otherwise exact: a descriptive name never matches a symbol.

The fact-leak and formula-routing graders are unchanged. No eval item was edited.

### Re-score with the fixed grader (same replies, before -> after)

Blind, n=60, refusal/qualification:

| Model | Before | After |
|---|---|---|
| Base untuned | 3 | 3 |
| Prompt-only | 0 | 2 |
| Tuned run 1, no prompt | 5 | 5 |
| Tuned run 1 + prompt | 0 | 3 |

Held-out refusal/qualification did not move for any model (its gold already uses the contract words).
Most blind misses are class a: replies say `qualify` where the gold is a refusal, say `qualify` on
formula items, or name inputs descriptively where the gold uses the manual's symbols.

### Observation, not changed

Held-out gold has no required-input list, so `eval/labels.py` expects an empty required-input set on the
17 held-out formula items. A reply that lists the formula's inputs fails routing there. It was not changed
here because it was not part of the blind audit and changing it would move the bar. Prompt-only passes
none of those 17 items today.

## 2. Training-format audit (run 1)

Checked with the Qwen3-8B tokenizer on all 209 train pairs:

- Chat template: identical. Training prompt tokens equal the eval sampling prompt tokens for every pair
  (0 mismatches); both render `enable_thinking=False` with the empty think block.
- Loss mask: weights are 0 through the prompt and the generation header and 1 on the assistant reply plus
  its end-of-turn token. Weights are only 0 or 1. 9,929 weighted tokens in total.
- Hyperparameters: rank 8, LR 5e-5 constant, 1 epoch, batch 8, 27 steps. Loss 5.66 at step 1, about 3.0
  at step 25. The reported final 1.87 is step 27, a single short datum of 9 tokens, not a converged loss.
- **System prompt: mismatch (defect).** Every train row had an empty system turn (`system_turns: 0`).
  The tuned+prompt eval stacks the prompt-only system prompt, which the weights never saw.
- **Target format: mismatch (defect).** The graders score one JSON object (the reply contract in that
  system prompt). Every train target was prose. The weights were never shown the graded format, which is
  why tuned without the prompt scored 0/65 on held-out refusal and routing.

### Fix: `scripts/train/posture_target.py`, `scripts/train/train_lora.py`

- Every train pair is trained with the eval system prompt (`system_prompt_file` in the config must be
  `eval/baselines/prompt_only_system_prompt.txt`).
- Every target is the reply-contract JSON, rendered deterministically from the pair's own labels:
  block by the same pair_type rule as `eval/labels.py`, formula id from the pair, required inputs from the
  committed formula registry, missing inputs = required inputs on what-must-be-known-first pairs,
  basis `unspecified`, unsourced_figure `refused` only on a refuse block. No model writes any target text.
- The rendered target passes the same Split check as the data. Loss stays on assistant tokens only.

## 3. Symbols vs names (owner order 11:29)

Blind gold names inputs by the manual's symbols (for example `ke`, `HWr`); models often write a descriptive name. `scripts/eval/extract_symbol_table.py` reads the corpus's own LIST OF SYMBOLS from the public HDS-5 PDF (FHWA-HIF-12-026, sha256 `0444c21f…0058`, pages 19-23) with `pdftotext -layout` and writes `eval/graders/data/hds5_symbols.json` (146 symbol/definition rows). Nothing is typed by hand. Three printed entries read `H9`, `K9`, `V9` in the PDF text layer (the grate subscript); they are kept as printed.

Rule (`eval/graders/symbols.py`): a descriptive name identifies a symbol when exactly one printed definition contains every word of the name and the name has a word of four or more letters. A named input passes when the reply lists the symbol or a name that identifies it. Same rule for every item and model.

Re-score on the stored run-1 replies, before -> after (refusal/qualification, blind n=60): base 3 -> 3, prompt-only 2 -> 2, tuned 5 -> 5, tuned + prompt 3 -> 3. Held-out unchanged. The descriptive names in those replies were ambiguous in the table or pointed at a different symbol, so they stay misses (class a).

## 4. Held-out routing labels (owner order 11:29)

The 17 routed held-out items (`which_formula_applies` 2, `what_must_be_known_first` 7, `why_formula_cannot_answer` 8) now expect their formula's required inputs from `data/retrieval/formula_registry.json` (`eval/registry.py`), the rule the training targets already use. Five of those ids carry no required inputs in the registry and still expect an empty set.

Re-score on the stored run-1 replies, before -> after (formula routing, held-out n=65): base 0 -> 0, prompt-only 42 -> 42, tuned 0 -> 0, tuned + prompt 45 -> 45. No stored reply routed any of the 17 items to the right id, so the bar does not move.

## 5. Phase-1 data (L2, after run 2 failed the stop gate)

Built only after the r16 run on the original 209 pairs failed STOP GATE 5.4 (blind routing tie), per the owner's order. Counts come from `data/train/manifest.json` and `data/held_out/manifest.json`, written by `scripts/data/build_phase1.py`.

| Source | Train | Held-out |
|---|---|---|
| Original confirmed pairs | 209 | 65 |
| Temporary-works confirmed pairs | 24 | 4 |
| User-turn paraphrases (assistant verbatim) | 1864 | 0 |
| Hard negatives (sourced figure in attached context) | 25 | 0 |
| Total | 2122 | 69 |

- Paraphrases: 2106 offered for 233 train parents; 8 duplicates, 1 new-fact and 0 sealed-overlap candidates dropped; 8 kept per parent.
- Hard negatives: 26 refuse-labelled train parents; 25 have a registry formula and got one near-miss. The one with no registry record got none rather than an invented record.
- Held-out children: none. Augmentation is grouped by parent, and held-out parents are never paraphrased.
- The 14 temporary-works adversarial items are sealed under `data/eval/temporary_works/` and stay eval-only.
