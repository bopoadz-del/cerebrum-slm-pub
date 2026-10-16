# Raw-line reconciliation

`scripts/data/reconcile_raw.py` traces every JSONL line in `data/raw/slm-datasets.zip` to exactly one destination: train, held_out, regression (sealed eval), retrieval, or dropped with a reason. Raw records carry no id field, so a line is identified by `member:line` and its content hash. `raw_lines.jsonl` is the full table (no item text). `summary.json` has the counts. `tests/data/test_reconcile_raw.py` fails CI if a line is missing or the table is stale.

796 JSONL lines: train 209, held_out 65, regression 61, retrieval 329 (5 fact-bearing pairs, 316 rate rows, 8 clause rows), dropped 132. The 76 YAML/JSON/TXT members are retrieval inputs or notes. Nothing is missing.

| Source member | Lines | Train | Held-out | Regression | Retrieval | Dropped |
|---|---|---|---|---|---|---|
| concrete/concrete_pairs | 28 | 24 | 4 | | | |
| concrete/concrete_adversarial | 12 | | | 12 | | |
| datacenter/datacenter_cases | 8 | | | 8 | | |
| facilities/unroutable | 1 | | | 1 | | |
| formulas/formula_selection (legacy tree) | 136 | 4 | | | | 132 |
| formulas/from-source/formula_selection | 132 | 107 | 20 | | 5 | |
| governance/governance | 27 | | 27 | | | |
| governance/governance_blind | 6 | | | 6 | | |
| mech-water/mech_water_epc | 26 | 22 | 4 | | | |
| mech-water/mech_water_epc_adversarial | 10 | | | 10 | | |
| plant/plant_fleet_maintenance | 33 | 28 | 5 | | | |
| plant/plant_fleet_adversarial | 14 | | | 14 | | |
| practice/clauses | 8 | | | | 8 | |
| rates/* market files (6) | 316 | | | | 316 | |
| rates/rate_handling | 4 | 3 | 1 | | | |
| site/site_pairs | 25 | 21 | 4 | | | |
| site/site_adversarial | 10 | | | 10 | | |

## Findings

- Governance: 27 + 6 = 33 lines. The 27 posture pairs are all in held-out, because governance is the held-out domain (rule: median explicit domain by count; `data/held_out/manifest.json` `held_out_domain`). The 6 `governance_blind` lines are sealed regression. None was collapsed by dedupe. Train shows governance = 0 by design.
- Datacenter: all 8 lines carry `meta.eval_only: true`, so they are sealed regression. Train shows datacenter = 0 by design.
- `candidate_exact_collapsed: 38`: all 38 are legacy-tree `formulas/formula_selection.jsonl` lines byte-identical (user + assistant) to a `from-source` line. No governance or other domain pair is involved.
- `formula_key_preferred: 94`: legacy-tree lines with the same formula id, pair type, and exact question as a `from-source` line, but older answer wording (for example units spelled into input names). The key includes the full question text, so it is not over-broad; restoring them would train two answers to one question. Nothing was restored.
- The 4 legacy-tree lines with no `from-source` twin are in train.

## Rows that do not come from the archive

`summary.json` `non_raw_rows` counts train and held-out rows built outside the archive: the owner's confirmed temporary-works pairs (`data/sources/temporary-works/`, 24 train, 4 held-out) and the phase-1 augmentation in train (1864 user-turn paraphrases, 25 hard negatives). They are traced to their parents by `scripts/gates/augment.py`, not by this table.

