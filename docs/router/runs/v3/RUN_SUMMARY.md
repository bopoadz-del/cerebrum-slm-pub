# Router run v3 — 2026-10-07 (Dubai, UTC+4)

Owner order: "train as is now". 21-label router adapter on the data merged in cerebrum-slm main at 0cccf02 (PR #37).

Data: data/router/train.jsonl (1482) and val.jsonl (848) from main at 0cccf02, used exactly as merged (no re-split, no new data). The merged rows were turned back into (label, excerpt) and rebuilt with the repo's train_router.training_examples (21-label prompt_for from configs/router/label_set_v8.json via router.labels; require_no_paths ran). Check: 1482/1482 and 848/848 rebuilt, system prompt equal to prompt_for on every row, user text unchanged on every row.
Split (from PR #37): v2 series-grouped stratified, seed 20261006; train caps drawing 400, other 300; val uncapped (drawing-heavy, 652/848).
Sealed: v5 and v8 were not read or scored. v8 is spent for the old adapter; a sealed score for this adapter needs a fresh v9 (not part of this run).

Model: Qwen/Qwen3.5-4B LoRA r16 (alpha 16), 2 epochs, lr 4.73e-4, batch 16, seed 0, 186 steps. Same recipe as run v2 except max_sequence_length (see walls).
Tinker training run: 34e98763-665a-534a-906f-651cc7e8c6f3:train:0
Epoch 1: tinker://34e98763-665a-534a-906f-651cc7e8c6f3:train:0/sampler_weights/router-v3-qwen3.5-4b-lora-r16-epoch1
Epoch 2: tinker://34e98763-665a-534a-906f-651cc7e8c6f3:train:0/sampler_weights/router-v3-qwen3.5-4b-lora-r16-epoch2
Parser: repo exact-alias parser (router.parse_label.parse_router_output via train_router._as_label). 0 unsure/off-list outputs in either epoch.

Val per-label recall (val n):   epoch1 / epoch2     predicted count e1/e2
RFI (7)                    0.143 / 0.00    pred 2/0
submittal (4)               1.00 / 1.00    pred 4/5
transmittal (1)             1.00 / 1.00    pred 1/1
VO/CO (1)                   1.00 / 1.00    pred 1/1
claim (0)                    n/a / n/a     pred 0/0
payment application (1)     1.00 / 0.00    pred 1/0
payment certificate (1)     1.00 / 1.00    pred 1/2
drawing (652)              0.986 / 0.979   pred 681/701
specification (3)          0.333 / 0.333   pred 1/2
schedule (8)                1.00 / 1.00    pred 10/17
report (43)                0.023 / 0.465   pred 1/22
correspondence (7)         0.143 / 0.143   pred 1/1
BOQ (2)                    0.500 / 1.00    pred 1/7
contract (11)               1.00 / 0.091   pred 12/1
method statement (3)        1.00 / 1.00    pred 3/3
meeting minutes (10)        1.00 / 1.00    pred 10/10
NCR (0)                      n/a / n/a     pred 0/3
inspection request (7)     0.143 / 0.571   pred 1/12
site instruction (8)        1.00 / 0.250   pred 75/6
register/log (3)           0.333 / 1.00    pred 1/3
other (76)                 0.447 / 0.224   pred 41/51
Macro recall over the 19 labels present in val: 0.687 / 0.635. Accuracy: 0.863 / 0.846.
Labels never predicted: epoch 1 claim, NCR; epoch 2 RFI, claim, payment application.
Majority baseline (always "drawing"): accuracy 0.769, macro 0.053, gate FAIL (as it should).
Model gate (majority/constant predictor + beats baseline macro): PASS both epochs.
Epoch 1 has the higher macro and accuracy. Epoch 2 shifts mass: report 0.023 -> 0.465 and inspection request 0.143 -> 0.571, but contract 1.00 -> 0.091 (predicted once), site instruction 1.00 -> 0.25 (epoch 1 over-predicts it, 75 times), RFI 0.143 -> 0.

Train label counts: {"transmittal": 12, "contract": 134, "other": 300, "drawing": 400, "correspondence": 71, "site instruction": 48, "report": 107, "BOQ": 73, "RFI": 29, "method statement": 29, "payment application": 4, "schedule": 49, "meeting minutes": 58, "VO/CO": 7, "specification": 66, "register/log": 27, "payment certificate": 7, "inspection request": 27, "submittal": 29, "NCR": 4, "claim": 1}

Walls:
- Tinker does not serve Qwen/Qwen3-4B. Used Qwen/Qwen3.5-4B, as in run v2.
- With the 21-label prompt (definitions from label_set_v8.json) 354 train rows exceeded max_sequence_length 3072 (longest 5979 tokens; longest val prompt 6062). Raised to 6144 for this run only; configs/lora is unchanged.
- train_router._encode on main already unwraps BatchEncoding; the driver still applies the run-v2 _encode_compat patch (no behaviour change).
- Val n is tiny for many labels (transmittal, VO/CO, payment application, payment certificate 1 each; BOQ 2; claim and NCR 0), so those recalls are noise. Macro covers 19 labels; claim and NCR are absent from val.
- Thin train labels (from PR #37): claim 1, payment application 4, NCR 4, VO/CO 7, payment certificate 7, transmittal 12.
- Run on the box (not ChaDs). Repo read from a read-only git archive of main at 0cccf02 made on ChaDs (box has no GitHub credentials); the archive has CRLF line endings, content identical to the PR #37 files.
- No commits. Nothing written to the repo.
