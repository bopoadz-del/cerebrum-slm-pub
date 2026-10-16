# Router v5 epoch 1: name-blind CHECK FIX (not a re-score)

Owner order 2026-10-07 22:20 +04 (B). PRE_RUN_NOTE.txt was written before the re-run.

- **Fixed:** `scripts/router/name_blind.py`, so that it blanks the party value only.
- **Re-run:** only the blanked side, ONCE, with the same adapter, prompt (system sha256 `4bb6ee17...dac56b`), sampling and parser.
- **Not re-run:** the unblanked side is the stored `docs/router/runs/v5/sealed/sealed_predictions.jsonl` from the one sealed score.
- **Unchanged:** no sealed score, macro, accuracy, gate_v2 or router_passed is recomputed. `docs/router/runs/v5/sealed/` stays the record of the sealed score and the first name-blind run.
- **Blanker sha256:** fixed `a235de3cbe5246b8cddc783bdfd0cc5d4ef6dae6dca6b04317de79b6e403e73c`; old `ddab8598...bdeabf`.

## Why: the dump of every flipped row

The classifier input was dumped before and after blanking for every flipped row: S5-021, S5-023, V8-029, V8-040, V8-113.
The combined correspondence flips are S5-021 and S5-023. The dump stays off the repo, and no document text is committed.
Extracted PDF text is one flat line, so the old 80-character window after a field colon ran past the party value:

| row | gold | old blanker also masked | verdict |
|---|---|---|---|
| S5-021 | correspondence | the salutation word, the next field labels (NAME OF, ADDRESS OF, NAME OF PAYMENT) | over-blanking |
| S5-023 | correspondence | the body sentence after the salutation, an instruction sentence, the salutation word, a label word after an empty form blank | over-blanking |
| V8-029 | register/log | the project title field (a listed title-block field) plus two words of the next unlabelled field | mostly a real title-block dependence |
| V8-040 | BOQ | the table header (ITEM NO.) and BOQ line items after `Project No:` | over-blanking |
| V8-113 | other | the document title, which sits in the `Issued by:` slot of a flattened title block, and a distribution table header | over-blanking (layout) |

## The fix (generic, by layout, no names)

- Leading form blanks stay, so an empty form field blanks nothing.
- A label ending in a number qualifier (No / Number / Ref / Reference) blanks one token, the code.
- The value ends before a run of two or more ALL-CAPS words (the next heading, table header or ALL-CAPS label) once a word with a lowercase letter has been seen.
- The value ends where running prose starts.
- A name before body text is still blanked. Capitals inside a name (an acronym, `X, LLC`) do not cut it.
- One rule was tried and rejected because it under-blanked real names: dropping `X by:` / `X to:` labels (Drawn by, Checked by, Issued to + a person or company).

On all 263 sealed rows the fixed blanker blanks 345 values (was 388). It masks 12,584 characters (was 17,275); 43 windows no longer blank anything (form blanks, prose, checkboxes).

## Result (stored unblanked predictions vs fixed-blanker predictions)

| set | n | rows blanked | fields | flips (passable / not) | flips on rows with nothing blanked | gate | old blanker flips (passable / not) |
|---|---:|---:|---:|---|---:|---|---|
| v5_retick | 64 | 39 | 106 | 2 (0 / 2) | 0 | PASS | 2 (0 / 2) |
| v8_retick | 185 | 89 | 214 | 2 (2 / 0) | 0 | FAIL: other, register/log | 3 (3 / 0) |
| cr_addendum | 14 | 11 | 25 | 0 (0 / 0) | 0 | PASS | 0 (0 / 0) |
| combined | 263 | 139 | 345 | 4 (4 / 0) | 0 | FAIL: correspondence, other, register/log | 5 (5 / 0) |

Flipped rows (unblanked -> old blanker -> fixed blanker):

- v5_retick S5-021 (gold correspondence): correspondence -> payment application -> payment application (5 fields)
- v5_retick S5-023 (gold correspondence): correspondence -> site instruction -> site instruction (1 fields)
- v8_retick V8-029 (gold register/log): transmittal -> other -> other (1 fields)
- v8_retick V8-040 (gold BOQ): schedule -> BOQ -> schedule (3 fields)
- v8_retick V8-113 (gold other): other -> payment certificate -> payment certificate (2 fields)

## Per-class cause

- **BOQ:** over-blanking. The old blanker masked the table header and BOQ line items after `Project No:`. Fixed: only the project number is masked now, and the flip is gone.
- **correspondence** (S5-021, S5-023; fails on combined): over-blanking of the salutation.
  - The fix removed the body-sentence and label overreach.
  - A one-word residual remains: the word just before a one-word Title-case label (`Dear` before `Sir:` / `Contractor:`).
  - In S5-023 that word is the ONLY thing masked, and the row still flips. So the cue is the salutation, not a party name.
  - This is not a party-name dependence and not a reason for rows. It needs one more generic rule, and the check was not re-run for it.
- **other** (V8-113): over-blanking caused by layout.
  - The flattened title block puts the document title in the `Issued by:` value slot and a table header in the `Issued to:` slot.
  - A layout-only rule cannot tell a title in a party slot from a company name. The rule that would remove it under-blanked real names, so it was rejected.
  - The structural remedy is layout-preserving text (line breaks) for the candidates, not rows.
- **register/log** (V8-029): mostly a real title-block dependence.
  - The masked value is the project title, a listed title-block field, plus two words of the next unlabelled field.
  - The unblanked prediction was already wrong (transmittal; gold register/log) and moves to other.
  - It is a weak class on this adapter (combined recall 0.267). This is the one class where more register/log rows from other projects is the indicated fix.

After this check fix the gate still FAILS on v8_retick (other, register/log) and on combined (correspondence, other, register/log). It passes on v5_retick and cr_addendum.
