# cerebrum-slm

Posture model for construction. The weights learn how to refuse, qualify, name a missing input, and route to a formula. Rates, clause text, formula values, brands, and site names stay in retrieval.

The retrieval source is The Fork. This repository does not add a Cerebrum Blocks Store block.

## Raw archive

`data/raw/slm-datasets.zip` is the dataset archive, committed byte for byte. `data/raw/SHA256SUMS` is its sha256 manifest. CI rejects a zip that does not match the manifest.

## Gates

CI (`.github/workflows/ci.yml`) runs on pull requests and on every branch push.

```bash
PYTHONPATH=scripts python -m gates.run
```

1. **No hardwiring.** `src/` may not contain a probe id or a domain name taken from the sealed catalog, and may not contain a word list. A training pair whose record hash, or whose normalized user/assistant hash, is in the sealed eval hashes fails. The catalog is built at run time from sealed eval records only: `data/eval/**`, and eval-only records in the raw archive (`*_adversarial`, `governance_blind`, `unroutable`, and records with `eval_only: true`). Training rows do not add names. A domain label that is a single word, such as `governance` or `datacenter`, is not a sealed-item name and is not banned. The match is case-sensitive, so a comment that does not repeat the sealed token does not fail. The catalog is not a list checked into `src/`. The content hash is the user and assistant text, so a schema rewrite that keeps that text still matches. A system prompt is not part of the hash. An `__all__` list, tuple, or set of Python identifiers is a module export, not a word list. A list assigned to a name such as `domains`, `brands`, `sites`, or `probe_ids` still fails, as does any other list or tuple of eight or more strings.

2. **The split.** A training assistant turn fails when it carries a number with a unit, a currency amount, a clause plus an edition, a brand, or a site name. A currency amount is a number with a code or symbol (`6 USD`, `SAR 12`, `$6`). A bare code with no number, including an input name such as `cement price sar t`, does not fail. A clause plus an edition is a clause or section with an edition or year, or a standard designation that carries an edition or a part (`ACI 318-19`, `EN 1992-1-1`). A bare standard number such as `BS 8110`, or a body name such as `CIRIA`, does not fail. Brands, site names, and clause titles are read from retrieval rows in the raw archive and from `data/retrieval/**`. A posture turn with none of those passes. A clause number with no edition passes.

3. **Blind-set seal.** Every file under `data/eval/**` is hashed. Each `*.json` or `*.jsonl` payload in that tree must match a committed seal in its directory: a `*.sha256` sidecar in sha256sum form (`digest  filename` of the raw bytes), or a `seal.json` with `algorithm`, `artifact`, and `sha256`. When `seal.json` includes `count` or `source_zip_sha256`, those must match the payload's non-empty line count and `data/raw/slm-datasets.zip`. A missing seal or a hash mismatch fails the build. A README is not part of the seal. Training code (`src/`, `scripts/train/`, and top-level `train*.py`) may not read that tree. The check parses Python and folds concatenated strings, joins, and path division, so spelling the path in pieces still fails. A read whose path is not a constant outside that tree must call `training_open`, which rejects `data/eval/**` before opening it. A shared sha256 between a training file and a blind file fails the build, so the training manifest and the blind manifest stay disjoint.

4. **Augmentation.** `scripts/gates/augment.py` checks every train row that carries `meta.aug`. Each one hashes to a parent pair in train (`meta.parent_id` is the parent's content hash, and its own `content_hash` matches its text). A child of a held-out parent never lands in train, and no parent sits in both splits. A paraphrase keeps the parent assistant turn byte for byte, keeps its pair type and formula id, and adds no number, unit, currency, clause, edition, brand, or site that the parent user turn did not carry. A hard negative routes to a formula whose id is in its attached Fork context and cites only ids in that context. No augmented user turn matches a sealed eval item by hash or reaches 0.8 token overlap (Jaccard over lower-cased word tokens) with one. The split check runs on augmented rows like on any training row.

Training files are `*.jsonl` and `*.json` under `data/`, except `data/raw/`, `data/eval/`, and `data/retrieval/`.

`gates.run` plants a probe id, a domain name, a word list, a sealed-hash copy, fact-bearing assistant turns, and a blind read in a temporary tree. Each plant is red, then removed, then green. Those files are not written into this repository.

## Phase-1 training data

`scripts/data/build_phase1.py` rebuilds `data/train/` and `data/held_out/` deterministically. Run it by hand and commit the output; CI does not generate pairs.

1. Parents. Confirmed posture pairs from the raw archive (through `apply_split`) plus `data/sources/temporary-works/temporary_works_pairs.jsonl` (confirmed, `eval_only: false`). Held-out is 15% per source group chosen by content hash, with the governance domain kept whole; this reproduces the earlier 65-item held-out and adds 4 temporary-works items.
2. Question-side children. `data/augment/user_paraphrases.jsonl` holds user-turn paraphrases of train parents only, generated by `scripts/data/paraphrase_users.py`. At most 3 are kept per parent after dropping duplicates, any new fact (number, unit, currency, clause, edition, brand, site, or a bare number not in the parent user turn), and anything at 0.8 token overlap or more with a sealed eval item. The assistant turn is the parent's, byte for byte.
3. Hard negatives. For each refuse-labelled train parent whose formula is in `data/retrieval/formula_registry.json`, one near-miss: a fixed user frame plus that registry record attached as Fork context with its id. The target is the reply contract rendered from labels (block none, that formula id, its required inputs). No free text is written on the assistant side.

`data/eval/temporary_works/temporary_works_adversarial_blind.jsonl` is eval-only and sealed. It is never trained on.

## LoRA entry point

`configs/lora/qwen3-8b-posture.json` is the posture adapter for `Qwen/Qwen3-8B`. Every train pair is trained with the eval system prompt and a reply-contract JSON target rendered from its labels (`scripts/train/posture_target.py`). `configs/lora/qwen3-8b-posture-r16.json` is the rank-16 run. `scripts/train/train_lora.py --dry-run` checks that config and the train split and prints counts. It does not contact Tinker. Hyperparameter reasoning is in `configs/lora/README.md`.

## Tuned eval

`scripts/eval/run_tuned_eval.py` samples a Tinker checkpoint of the Qwen3-8B LoRA on held-out and the blind set, grades with the refusal, fact-leak, and formula-routing graders, and writes counts only. `--dry-run` checks the model config, the held-out manifest, and the sealed blind file, and does not contact Tinker. `--prompt-only-system-prompt` stacks the committed prompt-only system prompt on the tuned checkpoint. Labels for both sets live in `eval/labels.py`, which the baseline entry points call without sampling again.

`scripts/eval/run_stop_gate.py` compares the **tuned + system prompt** aggregate with the prompt-only lines in `eval/baselines/scores.md`. Tuned-without-prompt is reported, not gated. The rule is in `eval/STOP_GATE.md`. FAIL means no serving. Routing samples attach the formula registry (held-out committed registry; blind HDS-5 equation list) with constrained decoding on formula ids; refusal/qualification and fact-leak stay context-free.

## What this entry point does not do

No training run is started by CI or by `--dry-run`. CI does not generate pairs. `data/eval/` stays sealed and is not read by the training entry point.

## Router labels

The document-type label set is `configs/router/labels.json`. The v5 sealed set is 64 owner-labelled public-web rows in `data/eval/router_unseen/`. The blank sheet remains `data/sealed/router/`. The raw Fork export is a document pool, not training labels (pull request #25). A teacher pass and a Qwen3.5-4B LoRA are recorded in `docs/router/DATASET_CARD.md`. Tinker does not serve `Qwen/Qwen3-4B`. Human agreement is done (0.875, kappa 0.864), and training proceeded before that check. The sealed set was scored once. sealed gate: FAIL (gate defect: condition (b) unsatisfiable as written); router NOT marked passed. `sealed_scoring` stays false. Sealed gate version 2 applies only to a future fresh sealed set. The v5 result is unchanged. See `docs/router/LABELS.md`.

## Local run

```bash
python -m pip install -r requirements.txt
PYTHONPATH=scripts python -m gates.run
PYTHONPATH=scripts python -m pytest tests
```


## Served artifact (epoch 2)

Stop-gate **PASS** under protocol v2. Blind refusal/qualification is **7/60** (see `KNOWN_LIMITATIONS.md`).

- Checkpoint: `tinker://d33e27e4-5533-5af4-9039-c9bf8254d4b7:train:0/sampler_weights/posture-qwen3-8b-lora-r16-epoch2`
- Config: `configs/serve/posture-epoch2.json`
- Client: `src/posture_model/` (system prompt ships with the artifact; Fork context kwarg required)
- Load/sample: `docs/serve/LOAD_AND_SAMPLE.md`
- Fork hook: `docs/serve/FORK_INTEGRATION.md`
- Example transcripts: `docs/serve/transcripts-epoch2.md`
