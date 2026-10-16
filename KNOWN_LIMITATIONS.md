Blind refusal/qualification on the sealed eval set is **7/60** (epoch 2 tuned + system prompt, protocol v2).

## Honest remaining limits (run 4, epoch 2, protocol v2)

These numbers are aggregate counts from `docs/runs/run4-cap3-weight-x3-stop-gate.md`. Item text is not repeated here.

| Surface | Score | Note |
| --- | --- | --- |
| Blind refusal/qualification | **7/60** | Worst honest number for this serve. Beats prompt-only 3/60; still far from reliable blind refusal. |
| Blind formula routing | **22/60** | Beats prompt-only 17/60; blind routing is not solved. |
| Held-out refusal/qualification | 62/69 | Passes the stop-gate vs prompt-only 51/69. |
| Held-out formula routing | 68/69 | Passes the stop-gate vs prompt-only 61/69. |
| Fact-leak (held-out / blind) | 0/69 / 0/60 | Required zero on both sets. |

## Serve caveats

- The served artifact is **tuned weights + the committed system prompt**. Sampling without that prompt is a diagnostic path only; the stop-gate does not treat it as a serve decision.
- Routing scores require an attached formula registry and constrained decoding on formula ids. Bare routing is not a valid score path and must not be used in Fork.
- Retrieval stays on The Fork. This repository does not embed rates, clause editions, brands, or site facts in the weights.
- Blind-set generalization remains weak on refusal. Do not claim production-grade blind refusal from 7/60.
- Do not add sealed eval item text to docs, transcripts, or logs.
