# Router dataset card

Status: thin-v9b adds VO/CO vs change-request contrastive pairs plus claim rows and re-runs the split (see the thin-v9b section). Before that, thin-v9 added public filled VO/CO, payment application, claim and RFI rows and re-runs the split (see the thin-v9 section). Before that, pr5 added the unmasked pr2 owner-source rows and v9 change-request rows and re-runs the split on the v9 label set (see the pr5 section). Earlier status: v8 21-label teacher relabel of the cleaned training pool, plus owner-laptop additions and a thin-label public/laptop top-up. Human agreement remains as recorded (0.875, kappa 0.864). Sealed v5 and sealed v8 are untouched. No training run in this PR.

Model version: Qwen/Qwen3-8B

The teacher is `Qwen/Qwen3-8B`, provider `tinker`, temperature `0`, max_tokens `128`. Relabel used the fixed confidence placeholder from main at 1a94dcf (PR #36): the prompt asks for a number between 0 and 1 instead of a literal `0`. On the 5627-row cleaned pool the teacher returned 0.95 on 5297 rows, so the 0.7 floor barely discriminates; treat confidence as a coarse filter.

The current builder in `scripts/router/teacher_label.py` loads definitions from `configs/router/label_set_v8.json`. The recorded template is:

```
You assign one document-type label from the document text only.
Choose exactly one label from the list, or unsure when the text does not support one label.
The list is the only label set. Do not add a label.
Use only the document text in the next message.
Do not use a filename, a folder, a document id, or a project name. They are not provided.
Reply with one JSON object and no other text:
{"label": "<one list entry or unsure>", "reason": "<one line>", "confidence": <your confidence, a number between 0 and 1>}
Allowed labels:
{label_list}
```

Labels come from document content only. Filename, folder, source path, source site and form number do not decide a label and do not appear in training text.

## thin-v9f: claim rows (label set v9, 22 labels)

- Why: claim was still thin at 19 train / 3 val after thin-v9b.
- `data/router/additions_thin_v9f.jsonl`: 8 rows, all labelled claim by the v9 teacher (Qwen/Qwen3-8B, temperature 0, 0.7 floor, current v9 prompt, text passed through `router.strip_source_tags`). They are contractor entitlement submissions from public FILLED documents: contract time-extension requests with particulars (delay causes, dates, days requested), a time-extension summary letter, and contractor claim/entitlement letters attached to agenda items. Sources: public county, school-district, airport-authority and town agenda packets (5 source groups). No court records are used. The teacher label is never changed; candidates the teacher labelled anything other than claim were dropped.
- Provenance: every row records `source_url`, `fetch_timestamp`, `http_status`, `content_sha256` and `page_title` (collection provenance rule), and the file is in the CI provenance check. These fields are row metadata only; the training text is the excerpt.
- Grouping: one project per source domain. A domain already used by thin-v9, thin-v9b or v9 change-request rows keeps that project id and split group.
- Isolation: sha, project, 12-word shingle, first-300 and source-domain checks against every sealed set under `data/eval/router_unseen` (7 files, found by glob) came back 0. Rows were also gated (hashes only, kept outside the repo) against all pending sealed candidates (thin-v9 b/c/d batches, the laptop batch e pool and the web_f batch): 0 hits.
- Split re-run with the same procedure (seed 20261006, drawing cap 400, other cap 300, templates train-only); with no new rows it reproduces main's train.jsonl and val.jsonl byte for byte. The sealed exclusion reads all 7 sealed files: 0 rejected. Train 1712 -> 1719, val 900 -> 901. Claim train 19 -> 25, val 3 -> 5. Whole source groups move between train and val in a grouped re-run; thin-v9b placement counts in `split_counts.json` are updated to match. `additions_thin_v9c.jsonl` and `additions_thin_d.jsonl` are not in train/val on main and stay out of this re-run, unchanged.

## thin-v9b: VO/CO vs change-request contrastive pairs and claim rows (label set v9, 22 labels)

- Why: on the v5 sealed score VO/CO was 0/16, with 15 rows predicted as change request. These pairs show the model the issued instrument and the request for the SAME change side by side.
- `data/router/additions_thin_v9b.jsonl`: 38 rows. 31 come from public FILLED packages (council and board agenda packets, procurement records) that hold both the request (COR, PCO, change proposal, request for change order) and the issued instrument (signed or issued change order, EJCDC C-941, AIA G701, contract change order) for the same change. Pairing comes from document lineage inside the package: the instrument cites the request by number, amount, or scope and date. No name list is used. Where the instrument was already a thin-v9 row, only the request is new, and the pair points to the existing row. The other 7 rows were collected as claim candidates: contractor entitlement submissions (notice of intent to claim, certified request for equitable adjustment, delay/acceleration claim letter, differing-site-condition REA, arbitration demand with claim items) from public county agenda packets and public court exhibits. The teacher labelled 6 of them claim and 1 other. The 'other' row is cut by the fixed-seed other cap of 300, so it is not in train/val. 2 more pages collected from those packets were labelled change request and VO/CO and form 2 further pairs.
- Labels come from content only: the v9 teacher (Qwen/Qwen3-8B, temperature 0, 0.7 floor) with the current v9 prompt from `prompt_for`, which includes the design rule "Directs the change = VO/CO, asks or prices the change = change request." (PR #49), on text passed through `router.strip_source_tags`. All 39 collected rows were labelled under this prompt. For the 26 pair rows collected before PR #49, labels and confidences match the earlier run under the prompt without that rule; no row changed. Every train/val system message is the current prompt. A row collected for one role is kept only when its content label supports that role: an instrument must be labelled VO/CO and a request change request. 1 row was dropped under this rule (a priced quotation with an acceptance block, labelled VO/CO). The teacher label is never changed.
- `data/router/vo_cr_pairs_v9b.json`: 19 complete pairs from 15 projects (16 pairs in train, 3 in val, by source group). Pair membership (`pair_id`, `pair_role`) is audit metadata only. It is not in train/val rows, prompts or text, and a test enforces this.
- Grouping: one project per source domain. A domain already used by thin-v9 or v9 change-request rows keeps that project id and split group, so a project never spans train and val.
- Isolation: sha, project, 12-word shingle and first-300 checks against sealed v5, v8, both v9 re-tick copies and the v9 change-request addendum all came back 0, and the 12-word gate dropped 0 rows. Source domains are disjoint from every sealed set and from the pending thin-v9 and change-request candidates. Some 12-word shingles are shared with pending thin-v9 candidates, all of them standard change-order form boilerplate (EJCDC C-941 / AIA G701).
- Split re-run with the same procedure (seed 20261006, drawing cap 400, other cap 300, templates train-only, 5 sealed exclusion files, 0 rejected): train 1681 -> 1712, val 894 -> 900. VO/CO train 39 -> 49, change request train 27 -> 45, claim train 14 -> 19; val VO/CO 7 -> 7, change request 5 -> 8, claim 2 -> 3. The 6 new claim rows come from 2 source groups (one county, one court-record host). Claim stays under 30 train.

## thin-v9: public filled VO/CO, payment application, claim and RFI rows (label set v9, 22 labels)

- `data/router/additions_thin_v9.jsonl`: 93 rows from public FILLED documents about the works (issued or signed change orders, variation orders, construction change directives and contract change orders; contractor pay applications with the certifier block blank; claims and extension-of-time letters with particulars; RFIs). No blank templates and no staff reports that only approve a document. 50 projects; at most about 25 rows from one document. Project ids are opaque (`th9_pNN`); one project per source domain, and each project is one split group. Rows from a domain already used by the v9 change-request rows reuse that row's group so a project never spans train and val (6 groups). Source sites are not stored; a domain hash is kept for the isolation test. Internal only.
- Labels come from content only: the v9 teacher (Qwen/Qwen3-8B, temperature 0, v9 prompt) at the 0.7 floor. No filename, folder, site or form number decides a label. Result: VO/CO 40 rows (26 projects), payment application 33 (20), RFI 9 (4), claim 5 (5), change request 4, correspondence 1, payment certificate 1.
- `router.strip_source_tags.strip_source_tags` is applied to every row. 3 rows held a file path; they were re-labelled on the stripped text (labels unchanged).
- Isolation gate: 100 collected rows; 7 dropped because they share a 12-word shingle with a sealed excerpt (form boilerplate). sha, project and first-300 checks against sealed v5, v8, both v9 re-tick copies and the v9 change-request addendum: 0 hits. Source domains are disjoint from every sealed set and from the pending candidates.
- Split re-run with the same procedure (CANARY drop, 0.7 floor, sealed exclusion now over 5 sealed files with 0 rows rejected, split_v2 grouped stratified, seed 20261006, drawing cap 400, other cap 300, templates train-only): train 1602 -> 1681, val 880 -> 894. Thin rows: 70 train, 23 val. Train counts: VO/CO 5 -> 39, payment application 9 -> 37, RFI 27 -> 37, claim 10 -> 14, change request 24 -> 27, payment certificate 11 -> 12, correspondence 79 -> 80, submittal 31 -> 30, NCR 19 -> 18; other labels unchanged. Val: VO/CO 1 -> 7, payment application 1 -> 6, RFI 7 -> 6, claim 1 -> 2, change request 4 -> 5, NCR 2 -> 3, submittal 4 -> 5. The extra groups shift group order and the capped samples, so 22 val rows moved to train, 15 train rows moved to val, and 272 capped `drawing`/`other` train rows changed membership.
- Short of the 30-train target: claim (14). Sealed claim candidates: 0.
- Sealed v9 thin-class candidates (55, from other projects and other domains) are with the owner for ticking. They are not in the sealed set and not in training. `data/router/thin_v9_sealed_candidates_isolation.json` holds only their hashes.

## pr7: encoded-bytes content rule

Text that is mostly encoded bytes has no content. `router.content.is_mostly_encoded` counts encoded runs: unbroken stretches of at least 64 characters from the base64 / hex / url-safe alphabet (ASCII letters, digits, `+ / = _ -`) with no whitespace. When those runs exceed 0.5 of the non-whitespace characters, the row is not content. `router.train_router.training_examples` skips such rows, so they reach neither train nor val. The rule is structural, with no ids, names, labels or lists. Words in any language, codes, numbers and URLs fall far below the threshold; the highest non-encoded row in the committed data scores 0.45.

Effect: 1 pool row dropped (an archive dumped as base64, label `other`, about 24k tokens). It was in train on a35e160 and in no val row. The split was re-run with the same seed and caps: train 1602, val 880, label counts unchanged. Because one pool row fewer shifts group order and the `other`-cap sample, 335 train rows and 14 val rows changed membership. Isolation: sealed rejected 0.

## pr5: unmasked pr2 owner sources and v9 change-request rows (label set v9, 22 labels)

Masking ruling (owner, 2026-10-07): the router adapter trains on unmasked rows. See `docs/audit/router_masking_ruling.md`. The name-blind check (`scripts/router/name_blind.py`) is the gate that replaces masking for the router.

- `data/router/additions_owner_sources_pr2.jsonl`: the 219 pr2 owner-source rows in their ORIGINAL unmasked text (196 positives, 23 blank-template negatives). 19 rows keep the label note `label_source: v8_definition_rule` with `teacher_label` for owner review. Templates are label `other`, train only, inside the other cap, never in val. Provenance: owner sources, internal only. Project ids are opaque (`pr2_pNN`).
- AAZ isolation result (recorded before this PR): 71 AAZ rows against sealed v5 (64) and v8 (185): sha hits 0, first-300 hits 0, 12-word shingle pairs 4 (at most 2 shared shingles each), max 5-shingle Jaccard 0.012. Isolated.
- v9 relabel applied to the pr2 rows by content (change-request type content -> change request; variation settlement recommendation -> report): 0 rows qualified, 0 label changes.
- `data/router/additions_change_request_v9.jsonl`: 26 rows from public FILLED change-request documents (COR, PCO, proposal requests, change proposals) about the works, from 15 projects. Labelled by the v9 teacher (Qwen/Qwen3-8B, temperature 0, v9 prompt) at the 0.7 floor: 25 change request, 1 VO/CO. One near-duplicate row dropped. Project ids are opaque (`cr9_pNN`); each project is one split group. Source sites are not stored; a domain hash is kept for the isolation test. Internal only.
- Sealed v9 change-request candidates (14, from other projects and other domains) are with the owner for ticking. They are not in the sealed set and not in training. `data/router/cr_v9_sealed_candidates_isolation.json` holds only their hashes.
- Isolation: sha + project against sealed v5, v8 and both v9 re-tick copies (rejected 0); 12-word shingles against sealed v8 excerpts: 0; change-request source domains are disjoint from every sealed set and from the pending candidates.

Split re-run on v9 labels, same procedure (CANARY drop, 0.7 floor, split_v2 series-grouped stratified, seed 20261006, drawing cap 400, other cap 300). Train 1602, val 880 (were 1482 / 848). pr2 rows in val: 0. Change-request rows: train 23, val 3 (new rows only).

| label | train before | train after | val before | val after |
| --- | ---: | ---: | ---: | ---: |
| RFI | 27 | 27 | 7 | 7 |
| submittal | 29 | 31 | 4 | 4 |
| transmittal | 12 | 12 | 1 | 1 |
| VO/CO | 5 | 5 | 0 | 1 |
| change request | 3 | 24 | 0 | 4 |
| claim | 1 | 10 | 0 | 1 |
| payment application | 4 | 9 | 1 | 1 |
| payment certificate | 7 | 11 | 1 | 2 |
| drawing | 400 | 400 | 652 | 658 |
| specification | 66 | 66 | 3 | 7 |
| schedule | 49 | 50 | 8 | 8 |
| report | 108 | 123 | 44 | 43 |
| correspondence | 71 | 79 | 7 | 10 |
| BOQ | 73 | 76 | 2 | 8 |
| contract | 134 | 136 | 11 | 11 |
| method statement | 29 | 27 | 3 | 5 |
| meeting minutes | 58 | 58 | 10 | 10 |
| NCR | 4 | 19 | 0 | 2 |
| inspection request | 27 | 48 | 7 | 7 |
| site instruction | 48 | 48 | 8 | 8 |
| register/log | 27 | 43 | 3 | 6 |
| other | 300 | 300 | 76 | 76 |

## Pool preparation

`data/raw/router/labels.jsonl` is unchanged in this PR. After empty-text and duplicate-doc drops: 5627 rows labelled. Isolation against sealed v5 and sealed v8 (sha + project) rejected 0.

Before this PR's relabel, 102 pool rows were removed for 12-word phrase overlap with sealed v8/v5 (including 72 DG2 Jacobs-copyright rows). The 6 synthetic CANARY_A/B fixture rows were dropped from the training pool. See `data/router/removed_overlap.json`.

## Teacher counts (21 labels, after overlap + canary drop, floor 0.7)

`data/router/teacher_labels.jsonl` holds the relabelled pool only (5482 rows). Owner-laptop and public thin-label additions are in `data/router/additions_owner_laptop.jsonl` with `provenance` set (`owner laptop, content-labelled` or `public web, content-labelled`). No paths. `project_id` is taken from content.

Pool kept at floor (by label):

| label | count |
| --- | ---: |
| drawing | 3951 |
| other | 771 |
| report | 147 |
| correspondence | 78 |
| meeting minutes | 68 |
| contract | 68 |
| specification | 63 |
| schedule | 57 |
| site instruction | 56 |
| BOQ | 43 |
| RFI | 36 |
| inspection request | 34 |
| register/log | 29 |
| submittal | 27 |
| method statement | 23 |
| transmittal | 13 |
| VO/CO | 8 |
| NCR | 4 |
| payment application | 3 |
| payment certificate | 2 |
| claim | 1 |

Combined pool + additions kept at floor: 5647 rows (see `data/router/split_counts.json`).

## Split v2 (proposed, 21 labels)

`scripts/router/split_v2.py` series-grouped stratified split, seed `20261006`. Drawing capped at **400** in train. **Other capped at 300** in train (sampled across projects, same helper). Val is left as the repo split produces it (drawing-heavy); macro recall is per-label, so that is fine.

Isolation: `router.isolation.require_isolated` on every example against sealed v5 and sealed v8. Path guard: `require_no_paths` on train and val.

| | rows |
| --- | ---: |
| train | 1482 |
| val | 848 |
| drawing before train cap | 3299 |
| other before train cap | 718 |

### Train label counts

| label | train | val |
| --- | ---: | ---: |
| RFI | 29 | 7 |
| submittal | 29 | 4 |
| transmittal | 12 | 1 |
| VO/CO | 7 | 1 |
| claim | 1 | 0 |
| payment application | 4 | 1 |
| payment certificate | 7 | 1 |
| drawing | 400 | 652 |
| specification | 66 | 3 |
| schedule | 49 | 8 |
| report | 107 | 43 |
| correspondence | 71 | 7 |
| BOQ | 73 | 2 |
| contract | 134 | 11 |
| method statement | 29 | 3 |
| meeting minutes | 58 | 10 |
| NCR | 4 | 0 |
| inspection request | 27 | 7 |
| site instruction | 48 | 8 |
| register/log | 27 | 3 |
| other | 300 | 76 |

### Thin labels (train &lt; 30)

RFI=29, submittal=29, transmittal=12, VO/CO=7, claim=1, payment application=4, payment certificate=7, method statement=29, NCR=4, inspection request=27, register/log=27

Full machine-readable summary: `data/router/split_counts.json`.

## Human agreement

Unchanged from the previous card (done; 0.875 / kappa 0.864). Sealed gate records unchanged. `sealed_scoring` stays false.

## Provenance notes

- Additions batch (145) and thin-label top-up from the owner's laptop: provenance `owner laptop, content-labelled`. Outside DG2/Master Folder. Per-project caps about 10.
- Public thin-label top-up: real project documents from NSIP published packs and other public archives. Domains already used for sealed v8 were excluded entirely. 12-word phrase check against sealed rows. Bot-check walls logged, never bypassed (see agent walls log).
- Nothing from sealed v5 or sealed v8 enters training.

## v9 web_f sealed addendum (2026-10-08)

- 40 public-web candidates were ticked by the owner. 34 are sealed as `data/eval/router_unseen/v9_web_f_addendum/sealed_labels.jsonl` and 6 unsure rows are held there in `held.jsonl`. None of the 40 is in train, val or any additions file. `data/router/web_f_sealed_candidates_isolation.json` holds only their hashes (status sealed or held), so they are kept out of later training batches.
- One held row (a meeting agenda) is an eval-only hard negative for meeting minutes. It never goes into training.
