# Router labels

The current label set is v9 in `configs/router/labels.json` (`label_set` is `v9`, 22 labels). Definitions are `docs/router/LABEL_SET_v9.md` and `configs/router/label_set_v9.json`. v9 adds one label to v8: `change request` (a scope change asked for or priced, not instructed). Code loads those files. It does not add a label. The v8 names are frozen in `configs/router/labels_v8.json` and its definitions in `configs/router/label_set_v8.json`. The v5 sealed set used the 12 labels frozen in `configs/router/labels_v5.json`, including `spec`. Replay of the recorded v5 and v8 scores loads those files and the original sealed rows, which are unchanged. The v9 re-tick of both sealed sets is a separate copy in `data/eval/router_unseen/v9_retick/` (see `RETICK.md`).

A label comes from document content. A filename, a folder, a document id, and a project id are not labels.

## Output aliases

The router output parser maps an exact case-insensitive alias onto one canonical label. The match is the whole string after whitespace is collapsed. There is no fuzzy match. A plural, a substring, and a near-miss stay unmatched. Conditional names (engineer's instruction, architect's instruction) are not aliases. `rfi` is not mapped to inspection request. Parsing rules are fixed now, before any v8 scores exist. The v9 aliases (change order request, variation request, potential change order, PCO, COR, request for modification, RFM, proposal request, variation proposal, change proposal → change request; construction change directive → VO/CO) are fixed before any v9 scores exist.

| alias | canonical |
| --- | --- |
| change order | VO/CO |
| variation order | VO/CO |
| variation | VO/CO |
| CO | VO/CO |
| VO | VO/CO |
| variation instruction | VO/CO |
| construction change directive | VO/CO |
| change order request | change request |
| variation request | change request |
| potential change order | change request |
| pco | change request |
| cor | change request |
| request for modification | change request |
| rfm | change request |
| proposal request | change request |
| variation proposal | change request |
| change proposal | change request |
| pay application | payment application |
| pay app | payment application |
| application for payment | payment application |
| interim application | payment application |
| progress claim | payment application |
| payment claim | payment application |
| IPC | payment certificate |
| interim payment certificate | payment certificate |
| final payment certificate | payment certificate |
| certificate of payment | payment certificate |
| spec | specification |
| request for inspection | inspection request |
| inspection and test request | inspection request |
| ITR | inspection request |
| WIR | inspection request |
| MIR | inspection request |
| work inspection request | inspection request |
| material inspection request | inspection request |
| register | register/log |
| log | register/log |
| notice of claim | claim |
| EOT claim | claim |
| request for information | RFI |
| non-conformance report | NCR |
| non-compliance report | NCR |
| bill of quantities | BOQ |
| schedule of rates | BOQ |
| minutes of meeting | meeting minutes |

The router model, the teacher, and a baseline share that parser (`scripts/router/parse_label.py`). The v5 sealed predictions were not re-parsed. The three v5 outputs that said `change order` stay recorded as misses.

## Raw pool

Pull request #25 (`data/router-labels`) is the raw document pool. That file is not training data. `fork_declared_type` is the stored Fork type, kept as a reference column. Nothing trains on it. Rows that still carry `fork_declared_type` or `declared_type` are unlabelled.

## Sealed eval

The v5 sealed set is 64 public-web documents, labelled by the owner. The blank 60-row sheet and the owner line stay in `data/sealed/router/`. The sealed labels are `data/eval/router_unseen/sealed_labels.jsonl` (`configs/router/sealed_eval.json`). `file_sha256` and `project` (`web_public` for this seal) are exclusion keys, not labels. The sealed set was scored once under sealed gate version 1. `sealed_scoring` stays false. sealed gate: FAIL (gate defect: condition (b) unsatisfiable as written); router NOT marked passed. Version 2 in `docs/router/SEALED_GATE.md` applies only to a future fresh sealed set. The v5 predictions were not re-scored or re-graded.

Holding out one Fork project is an extra in-pool check only. It is not the sealed set.

A training split refuses a pool row whose document sha256 or project is already in `data/eval/router_unseen/sealed_labels.jsonl`. The training entry point does not read `data/eval/`.

## Training path

`scripts/router/teacher_label.py` labels a pool from content when a key is set. It still refuses `data/sealed/` and `data/eval/router_unseen/`. An empty sealed file does not block the pool. `scripts/router/human_sample.py` draws a stratified sheet from content labels that already exist, including rare labels, without cloning rows. `scripts/router/report_counts.py` counts those labels. `scripts/router/majority_gate.py` fails a predictor that returns the majority label for every row, including when that predictor's accuracy is 1.

Human agreement is done (0.875, kappa 0.864). `data/router/human_sample_PENDING_OWNER.csv` is still a header only. By owner order, training proceeded before that check. The override is recorded in `configs/router/train.json`. It does not change the agreement gate.

The fixed teacher is `Qwen/Qwen3-8B` at temperature 0. The laptop run labelled the pool and trained a LoRA. Tinker does not serve `Qwen/Qwen3-4B`, so the adapter is `Qwen/Qwen3.5-4B`, rank 16, two epochs, 168 steps. The checkpoint is in `configs/serve/router-epoch2.json`. Split v2 (`scripts/router/split_v2.py`) groups by document series because the pool has no folder path, and caps drawing at 600 in train only. Human agreement is done (0.875, kappa 0.864). The sealed set was scored once and the router is not marked passed. Counts are in `docs/router/DATASET_CARD.md`.
