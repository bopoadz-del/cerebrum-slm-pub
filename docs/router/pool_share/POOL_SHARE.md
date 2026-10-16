# Router training pool: practice-region share (read-only)

Pool: `data/router/teacher_labels.jsonl` (4555 rows, sha256 8f322f29…4b82, excerpts present).
Script: `scripts/router/pool_region_share.py`. Result: `docs/router/pool_share/pool_region_share.json`.
No labels, sealed data or training files were changed.

## Method
Content only: the `excerpt` text. Embedded `[source: <path>]` tags are stripped first.
Filename, folder, project_id, source site and the teacher label are never inputs
(the teacher label appears in the result only as a reporting breakdown).
Signal families (regex, counted once per document), weights strong 3 / medium 2 / weak 1:
- a (US): AIA/G70x/A20x/ConsensusDocs/EJCDC (3); DOT/FHWA/OSHA/USACE (3); US state + ZIP (3); US state names (2);
  "change order" with Owner/Architect (2); punch list/retainage/schedule of values/pay app/NTP (2); county/city of (1);
  imperial units (1); $ amounts (1); US spelling (1).
- b (FIDIC/Gulf/UK/Commonwealth): FIDIC/NEC/JCT/RIBA/AS-NZS (3); Gulf places/authorities (3); UK/AU/NZ places (3);
  SAR/AED/OMR/QAR/KWD/BHD/GBP/AUD/NZD/£/riyal/dirham amounts (3); BS/BS EN/SMM7/NRM/POMI (2);
  the Engineer/Employer/QS/Contract Administrator (2); VO/IPC/sub-clause/EOT/letter of acceptance/progress claim (2);
  UK spelling incl. programme/metre/tonne (1); metric units (1).
- Rule: a if score_a >= 3 and score_a >= score_b + 2; b if score_b >= 3 and score_b >= score_a + 2; else c.
  Spelling or units alone never classify a row.

## Result
| class | rows | share |
|---|---|---|
| a US-form / US practice | 0 | 0.00% |
| b FIDIC / Gulf / UK / Commonwealth | 4120 | 90.45% |
| c indeterminate | 435 | 9.55% |

b >= one third of the pool: yes.

Caveats: b is dominated by one Saudi project's drawings (3577 of 4120 b rows are teacher-labelled
drawing). Excluding drawings: 863 rows, b 543 (62.9%), c 320 (37.1%), a 0. Explicit FIDIC/NEC/JCT
regime text appears in 2 rows; any contract-practice signal (regime, Engineer/Employer/QS roles,
VO/IPC/sub-clause/EOT, Gulf/UK/AU currency amounts) in 183 rows (4.0%). So the pool is regionally
Gulf but thin on FIDIC-style contract instruments (VO, IPC, claims, payment applications).

## Spot-check (seed 20261007, 25 per class)
- b: 25/25 read as Gulf content (Diriyah Gate II / Riyadh / Dubai). No false positives seen.
- c: 16/25 carry same-project Gulf context without a cue the rule counts; 9/25 have no regional cue; 0 US.
- a: empty. All 18 rows with US score >= 2 were read: OSHA 29 CFR guidance, an ASCE 7-16 excerpt,
  an AU utility bill, Gulf documents with US spelling. None is US-form contract paperwork.
