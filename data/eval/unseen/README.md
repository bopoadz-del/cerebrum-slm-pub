# Unseen eval

Shape-graded questions over one public corpus. Wording is not graded.

## Corpus

Hydraulic Design of Highway Culverts, Hydraulic Design Series Number 5, Third Edition (April 2012).

- Publication: FHWA-HIF-12-026
- Publisher: Federal Highway Administration, U.S. Department of Transportation
- Public URL: https://www.fhwa.dot.gov/engineering/hydraulics/pubs/12026/hif12026.pdf

This is a U.S. government work. Pilot documents were not used. Formula ids below are the equation numbers printed in that edition.

## Files

- `questions.jsonl` — 60 questions, one JSON object per line.
- `questions.jsonl.sha256` — SHA-256 of the raw bytes of `questions.jsonl`. That digest is the seal. This README is not part of the seal.

## Grade contract

Score a response correct only when all three shape fields match the gold object `shape`:

| Field | Match |
| --- | --- |
| `expected_block` | exact string |
| `formula_id` | exact string, or JSON `null` |
| `named_inputs` | equal as a set; gold stores them sorted |

Do not score prose, numeric results, or `source_locator`.

`expected_block` is one of:

- `formula` — one equation applies. `formula_id` is that equation. `named_inputs` is the full binding for this question.
- `missing_input` — one equation would apply, and at least one required symbol is absent. `formula_id` is that equation. `named_inputs` lists only the missing symbols.
- `qualifier` — a documented switch is unset, so no single equation is determined. `formula_id` is null. `named_inputs` names that switch.
- `false_premise` — the question asserts something the manual contradicts. `formula_id` is the equation that assertion falsely licenses. `named_inputs` is the binding it treats as valid. Do not compute a value.
- `refusal` — the requested object is not a formula result in this manual. `formula_id` is null. `named_inputs` names the unsupported object.

Counts: 20 `refusal_false_premise` (10 `false_premise`, 10 `refusal`), 20 `qualifier_missing_input` (10 `qualifier`, 10 `missing_input`), 20 `formula_routing` (block `formula`).

U057 and U060 share one shape on purpose. They are two wordings of submerged inlet control, Equation A.3. Every other shape is unique.

## Formula ids

`HDS5-3.1`, `HDS5-3.2`, `HDS5-3.3`, `HDS5-3.4a`, `HDS5-3.4b`, `HDS5-3.4c`, `HDS5-3.4d`, `HDS5-3.4e`, `HDS5-3.5`, `HDS5-3.6a`, `HDS5-3.6b`, `HDS5-3.7`, `HDS5-3.8`, `HDS5-3.9`, `HDS5-A.1`, `HDS5-A.2`, `HDS5-A.3`, `HDS5-A.4`.
