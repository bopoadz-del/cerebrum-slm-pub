# Fork dev integration (thin hook)

Owner rule: retrieval is The Fork. Do **not** add `retrieval_core` to Cerebrum-Blocks. Prefer this cerebrum-slm posture block + a thin Fork call site. No hardwiring of sealed probe ids or domain word lists into `src/`.

## Intended call order (dev)

1. **Retrieve** — Fork `app.core.rag.retriever.retrieve(query, project_id, ...)`.
2. **Attach** — render chunk / clause / rate text as `fork_context` (see `src/retrieval/context.py` and `src/retrieval/fork.py`).
3. **Posture sample** — `posture_model.sample_posture(user_text, fork_context=..., allowed_formula_ids=...)`.
4. **Formula** — if posture returns a `formula_id` present in the attached registry and inputs are complete, call `app.agents.formulas.run_formula`.
5. **Ground** — `src.retrieval.grounding.grounding_gate(proposed, context, formula_result=...)` before accepting any figure-bearing final answer.
6. **Escalate** — `src.policy.escalation.decide(...)` when confidence is low or context is absent.

## Client stub (Fork side, not committed here)

```python
# Sketch only — live in The Fork when a product PR is approved.
from posture_model import sample_posture

def posture_final_answer(user_text: str, fork_context: str, formula_ids: set[str]) -> str:
    return sample_posture(
        user_text,
        fork_context=fork_context,
        allowed_formula_ids=formula_ids,
    )
```

Environment: set `TINKER_API_KEY` in the Fork dev process. Point optional config `POSTURE_CHECKPOINT` at the pinned path only if overriding the committed serve config (default is the epoch-2 path in `configs/serve/posture-epoch2.json`).

## Fork PR wall

This cerebrum-slm change ships the interface and docs. Opening a product PR against `bopoadz-del/The_Fork` requires a clean worktree on that repo and an explicit owner go-ahead for the chat/runtime call site. If that PR is not opened in this lane, Fork continues to call its existing LLM path until the stub above is wired behind a feature flag.

## Cerebrum-Blocks

Do not register a Store block that embeds retrieval. If a Store wrapper is needed later, it must call this posture client and refuse when Fork context is missing — same contract as `sample_posture`.
