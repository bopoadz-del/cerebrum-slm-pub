#!/usr/bin/env python3
"""Generate user-turn paraphrases for confirmed train parents (question side only).

Input: the train parents ``build_phase1.py --emit-parents`` writes (parent id and
user turn only). Output: ``data/augment/user_paraphrases.jsonl``, one row per
parent with the raw candidate variants. No assistant turn is shown to or written
by the generator. ``build_phase1.py`` filters the candidates (new facts, sealed
overlap, duplicates) and copies each parent's assistant turn verbatim.

The generator is a Tinker-hosted instruct model. The API key is read from the
environment only.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GENERATOR = "Qwen/Qwen3.6-27B"
CANDIDATES = 10
INSTRUCTION = (
    "Rewrite the question below in {k} different ways that a construction engineer might ask it. "
    "Keep exactly the same meaning and the same request. Vary the vocabulary, the sentence length, and the word "
    "order: some short and blunt, some longer and more formal, some as a statement of the situation followed by "
    "the ask. If the question mentions a unit, you may write that unit another way (spelled out or abbreviated), "
    "but keep every number exactly as given. Do not add any number, unit, currency, price, standard, code, clause, "
    "edition, brand, product, company, place, site, or project name, and do not add any fact that is not already "
    "in the question. Do not answer the question. Return only a JSON array of {k} strings."
)
_ARRAY = re.compile(r"\[.*\]", re.S)


def parse_variants(text: str) -> list[str]:
    match = _ARRAY.search(text)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=GENERATOR)
    parser.add_argument("--k", type=int, default=CANDIDATES)
    args = parser.parse_args()

    import tinker
    from tinker import types

    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=args.model)
    tokenizer = client.get_tokenizer()
    params = types.SamplingParams(max_tokens=1500, temperature=0.9, top_p=0.95, seed=0)
    parents = [json.loads(line) for line in args.parents.read_text(encoding="utf-8").splitlines() if line.strip()]

    def one(parent: dict) -> dict:
        messages = [
            {"role": "user", "content": INSTRUCTION.format(k=args.k) + "\n\nQuestion:\n" + parent["user"]},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        variants: list[str] = []
        for attempt in range(3):
            result = client.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=params).result()
            reply = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
            reply = re.sub(r"(?s)<think>.*?</think>", "", reply)
            variants = parse_variants(reply)
            if variants:
                break
        return {"parent_id": parent["parent_id"], "generator": args.model, "variants": variants}

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(one, parents))
    rows.sort(key=lambda row: row["parent_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"parents": len(rows), "empty": sum(1 for row in rows if not row["variants"])}))


if __name__ == "__main__":
    main()
