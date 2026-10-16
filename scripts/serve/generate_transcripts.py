"""Generate ten synthetic serve transcripts from the pinned epoch-2 artifact.

Prompts are synthetic. They do not copy sealed eval text or held-out pairs.
Requires TINKER_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_ROOT, _ROOT / "src", _ROOT / "scripts"):
    _text = str(_entry)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from posture_model import load_serve_artifact, sample_posture
from posture_model.http_sample import sample_messages_http

# Synthetic scenarios only. Never copy data/eval/** or sealed item text.
SCENARIOS: list[dict[str, object]] = [
    {
        "id": "syn-01-refuse-unsourced-rate",
        "kind": "refuse",
        "user": "What is the unit rate for reinforced concrete grade C40/50 on this job?",
        "fork_context": "",
        "allowed_formula_ids": frozenset(),
    },
    {
        "id": "syn-02-qualify-missing-span",
        "kind": "qualify",
        "user": "Check cantilever deflection under uniform load. I have w, E, and I, but not the span.",
        "fork_context": (
            "Formula registry (id, name, inputs).\n"
            "- id: beam_deflection_cantilever_udl | name: cantilever UDL deflection | "
            "inputs: L, w, E, I"
        ),
        "allowed_formula_ids": frozenset({"beam_deflection_cantilever_udl"}),
    },
    {
        "id": "syn-03-route-cantilever",
        "kind": "route",
        "user": "Route the cantilever beam UDL deflection formula for this check.",
        "fork_context": (
            "Formula registry (id, name, inputs).\n"
            "- id: beam_deflection_cantilever_udl | name: cantilever UDL deflection | "
            "inputs: L, w, E, I\n"
            "- id: beam_deflection_simply_supported_udl | name: simply supported UDL deflection | "
            "inputs: L, w, E, I"
        ),
        "allowed_formula_ids": frozenset(
            {"beam_deflection_cantilever_udl", "beam_deflection_simply_supported_udl"}
        ),
    },
    {
        "id": "syn-04-refuse-invented-clause",
        "kind": "refuse",
        "user": "Quote the edition year for the temporary works code that applies here.",
        "fork_context": "",
        "allowed_formula_ids": frozenset(),
    },
    {
        "id": "syn-05-qualify-basis-unspecified",
        "kind": "qualify",
        "user": "Is the steel grade as-designed or as-built for this connection?",
        "fork_context": "",
        "allowed_formula_ids": frozenset(),
    },
    {
        "id": "syn-06-route-with-inputs-named",
        "kind": "route",
        "user": "I need the simply supported UDL deflection route; list required inputs only.",
        "fork_context": (
            "Formula registry (id, name, inputs).\n"
            "- id: beam_deflection_simply_supported_udl | name: simply supported UDL deflection | "
            "inputs: L, w, E, I"
        ),
        "allowed_formula_ids": frozenset({"beam_deflection_simply_supported_udl"}),
    },
    {
        "id": "syn-07-refuse-brand-from-memory",
        "kind": "refuse",
        "user": "Which brand of formwork system did this site use last year?",
        "fork_context": "",
        "allowed_formula_ids": frozenset(),
    },
    {
        "id": "syn-08-qualify-missing-load",
        "kind": "qualify",
        "user": "Run cantilever UDL deflection. Span L=3.0 m, E and I known; load w is unknown.",
        "fork_context": (
            "Formula registry (id, name, inputs).\n"
            "- id: beam_deflection_cantilever_udl | name: cantilever UDL deflection | "
            "inputs: L, w, E, I"
        ),
        "allowed_formula_ids": frozenset({"beam_deflection_cantilever_udl"}),
    },
    {
        "id": "syn-09-refuse-site-fact",
        "kind": "refuse",
        "user": "How many tower cranes are on the north plot today?",
        "fork_context": "",
        "allowed_formula_ids": frozenset(),
    },
    {
        "id": "syn-10-route-reject-outside-id",
        "kind": "route_constrained",
        "user": "Route using whatever formula id you prefer for deflection.",
        "fork_context": (
            "Formula registry (id, name, inputs).\n"
            "- id: beam_deflection_cantilever_udl | name: cantilever UDL deflection | "
            "inputs: L, w, E, I"
        ),
        "allowed_formula_ids": frozenset({"beam_deflection_cantilever_udl"}),
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Transcript jsonl path. Defaults to SLM_TRANSCRIPTS_JSONL. The in-repo path is not written.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=_ROOT / "docs" / "serve" / "transcripts-epoch2.md",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        override = os.environ.get("SLM_TRANSCRIPTS_JSONL", "").strip()
        if not override:
            print(
                "private data is absent: docs/serve/transcripts-epoch2.jsonl. "
                "Set SLM_TRANSCRIPTS_JSONL or pass --output. "
                "Refusing to write transcripts into the tree.",
                file=sys.stderr,
            )
            return 2
        args.output = Path(override)
    artifact = load_serve_artifact()
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        allowed = scenario["allowed_formula_ids"]
        assert isinstance(allowed, frozenset)
        reply = sample_posture(
            str(scenario["user"]),
            fork_context=str(scenario["fork_context"]),
            allowed_formula_ids=allowed if allowed else None,
            artifact=artifact,
            # Prefer OAI HTTP on this host: native Tinker SDK may be blocked (pyqwest/App Control).
            sampler=lambda msgs, _art=artifact: sample_messages_http(msgs, artifact=_art),
        )
        rows.append(
            {
                "id": scenario["id"],
                "kind": scenario["kind"],
                "checkpoint": artifact.checkpoint,
                "system_prompt": "prompt_only",
                "user": scenario["user"],
                "fork_context_attached": bool(str(scenario["fork_context"]).strip()),
                "allowed_formula_ids": sorted(allowed),
                "assistant": reply,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        "# Ten synthetic serve transcripts (epoch 2)",
        "",
        "Generated from the pinned serve artifact. Prompts are synthetic.",
        "No sealed eval item text. No held-out pair text.",
        "",
        f"- Checkpoint: `{artifact.checkpoint}`",
        f"- System prompt: `{artifact.system_prompt_file}`",
        f"- Count: {len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['id']}")
        lines.append("")
        lines.append(f"- kind: `{row['kind']}`")
        lines.append(f"- fork_context_attached: `{row['fork_context_attached']}`")
        lines.append("")
        lines.append("**User**")
        lines.append("")
        lines.append(str(row["user"]))
        lines.append("")
        lines.append("**Assistant**")
        lines.append("")
        lines.append("```json")
        try:
            pretty = json.dumps(json.loads(str(row["assistant"])), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pretty = str(row["assistant"])
        lines.append(pretty)
        lines.append("```")
        lines.append("")
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
