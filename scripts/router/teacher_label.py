"""Teacher labelling from document content. Dry-run writes nothing.

The system prompt is built at call time from configs/router/label_set_v9.json.
Definitions, "Is not" lines, regional names, notes, and the design rules are
not copied into this module. The allowed label names are read from
configs/router/labels.json at call time. Unsure and low-confidence rows are
not training rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from router.content import model_user_text
from router.guard import RouterWall, refuse_sealed_input
from router.labels import confidence_floor, load_label_set, load_labels, load_sealed_config, repo_root
from router.parse_label import canonical_label
from router.sheet import UNSURE

_CARD = Path("docs/router/DATASET_CARD.md")
_MODEL_LINE = re.compile(r"^Model version:.*$", re.MULTILINE)

_PROMPT = """You assign one document-type label from the document text only.
Choose exactly one label from the list, or unsure when the text does not support one label.
The list is the only label set. Do not add a label.
Use only the document text in the next message.
Do not use a filename, a folder, a document id, or a project name. They are not provided.
Reply with one JSON object and no other text:
{{"label": "<one list entry or unsure>", "reason": "<one line>", "confidence": <your confidence, a number between 0 and 1>}}
Design rules:
{design_rules}
Allowed labels:
{label_list}
"""

_KEY_ENV = ("TINKER_API_KEY", "ROUTER_TEACHER_API_KEY")


def prompt_for(
    labels: list[str] | None = None,
    *,
    label_set: dict | None = None,
    path: Path | None = None,
) -> str:
    """List each label's approved definition and "Is not" line.

    Regional names and notes are included when the label-set file has them.
    The design rules are copied from that file verbatim. Pass ``label_set``
    or ``path`` in tests; otherwise the approved file is read at call time.
    """
    data = label_set if label_set is not None else load_label_set(path)
    entries = [
        item
        for item in data["labels"]
        if isinstance(item, dict) and isinstance(item.get("label"), str)
    ]
    if labels is None:
        selected = entries
        bare: list[str] = []
    else:
        wanted = set(labels)
        selected = [item for item in entries if item["label"] in wanted]
        known = {item["label"] for item in selected}
        bare = [label for label in labels if label not in known]
    blocks = [_label_block(item) for item in selected]
    blocks.extend(f"- {label}" for label in bare)
    rules = [rule for rule in data["design_rules"] if isinstance(rule, str)]
    return _PROMPT.format(
        design_rules="\n".join(f"- {rule}" for rule in rules),
        label_list="\n".join(blocks),
    )


def _label_block(item: dict) -> str:
    lines = [
        f"- {item['label']}",
        f"  Definition: {item['definition']}",
        f"  Is not: {item['is_not']}",
    ]
    regional = item.get("regional_names_text")
    if isinstance(regional, str) and regional.strip():
        lines.append(f"  Regional names that map here: {regional}")
    notes = item.get("notes")
    if isinstance(notes, list):
        for note in notes:
            if isinstance(note, str) and note.strip():
                lines.append(f"  {note}")
    return "\n".join(lines)


def messages_for(row: dict, labels: list[str]) -> list[dict[str, str]] | None:
    """Build the teacher messages from content only. Returns None without content."""
    excerpt = model_user_text(row)
    if excerpt is None:
        return None
    return [
        {"role": "system", "content": prompt_for(labels)},
        {"role": "user", "content": excerpt},
    ]


def parse_teacher_response(text: str, labels: list[str]) -> dict:
    """One label from the file, or unsure. Anything else becomes unsure.

    An exact case-insensitive alias is stored as its canonical label. The
    router model and a baseline use the same map via ``canonical_label``.
    """
    payload = _json_object(text)
    raw_label = payload.get("label")
    matched = canonical_label(raw_label, labels) if isinstance(raw_label, str) else None
    if raw_label == UNSURE or matched is None:
        label = UNSURE
    else:
        label = matched
    reason = payload.get("reason")
    if not isinstance(reason, str):
        reason = ""
    reason = " ".join(reason.split())
    confidence = _confidence(payload.get("confidence"))
    return {"label": label, "reason": reason, "confidence": confidence}


def eligible_for_train(
    row: dict,
    labels: list[str],
    floor: float | None,
) -> bool:
    """Unsure, unknown labels, missing confidence, and below-floor rows stay out."""
    label = row.get("label")
    if label not in set(labels):
        return False
    confidence = row.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    if floor is None:
        return False
    return float(confidence) >= floor


def filter_for_train(
    rows: list[dict],
    labels: list[str],
    floor: float | None,
) -> list[dict]:
    return [row for row in rows if eligible_for_train(row, labels, floor)]


def label_rows(rows: list[dict], labels: list[str], respond) -> list[dict]:
    """Call respond(messages) for each content row. respond is injected by tests.

    The result keeps doc_id and project_id so a later split can group projects.
    Those fields are not sent to the teacher. Filename and Fork type are not copied.
    """
    labelled: list[dict] = []
    for row in rows:
        messages = messages_for(row, labels)
        if messages is None:
            continue
        parsed = parse_teacher_response(respond(messages), labels)
        if isinstance(row.get("doc_id"), str):
            parsed["doc_id"] = row["doc_id"]
        if isinstance(row.get("project_id"), str):
            parsed["project_id"] = row["project_id"]
        sha = row.get("content_sha256")
        if isinstance(sha, str) and sha.strip():
            parsed["file_sha256"] = sha.strip()
        excerpt = model_user_text(row)
        if excerpt is not None:
            parsed["excerpt"] = excerpt
        labelled.append(parsed)
    return labelled


def with_model_version(card: str, model: str) -> str:
    """Record a teacher model version on the dataset card text. Does not read env."""
    cleaned = model.strip()
    if not cleaned or any(char in cleaned for char in "\n\r"):
        raise RouterWall("WALL: model version missing")
    if not _MODEL_LINE.search(card):
        raise RouterWall("WALL: dataset card has no model line")
    return _MODEL_LINE.sub(f"Model version: {cleaned}", card, count=1)


def key_is_set() -> bool:
    """True when a teacher key is present. The value is never read out."""
    return any(bool(os.environ.get(name)) for name in _KEY_ENV)


def dry_run_report(rows: list[dict], labels: list[str]) -> dict:
    content_rows = sum(1 for row in rows if messages_for(row, labels) is not None)
    return {
        "status": "dry-run",
        "content_rows": content_rows,
        "labelled": 0,
        "wrote": False,
        "api_key_present": key_is_set(),
    }


def _load_input(path: Path | None, root: Path) -> list[dict]:
    if path is None:
        return []
    refuse_sealed_input(path, root)
    if not path.is_file():
        return []
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RouterWall(f"WALL: {path}:{lineno}: {exc}") from exc
        if not isinstance(obj, dict):
            raise RouterWall(f"WALL: {path}:{lineno}: record is not an object")
        rows.append(obj)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Teacher-label router documents from content")
    parser.add_argument("--input", type=Path, default=None, help="JSONL pool. Omit for a 0-row dry run.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Label the pool from content. Does not wait for a sealed sheet. Refuses the sealed tree.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    try:
        labels = load_labels(root / "configs" / "router" / "labels.json")
        config = load_sealed_config(root / "configs" / "router" / "sealed_eval.json")
        rows = _load_input(args.input, root)
        if args.write:
            _refuse_pool_write(root, args.input)
            report = write_pool_labels(root, rows, labels, config, args.output)
        else:
            report = dry_run_report(rows, labels)
            report["confidence_floor"] = confidence_floor(config)
            report["teacher_model"] = config["teacher"]["model"]
    except RouterWall as exc:
        print(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def _refuse_pool_write(root: Path, input_path: Path | None) -> None:
    """Pool labelling does not wait for the sealed file. The sealed set itself is refused."""
    if input_path is None:
        raise RouterWall("WALL: no pool to label")
    refuse_sealed_input(input_path, root)
    if not key_is_set():
        raise RouterWall(
            "WALL: TINKER_API_KEY is not set; the owner key file was not reachable "
            "and the environment has no key. Teacher labelling did not run."
        )


def write_pool_labels(root: Path, rows: list[dict], labels: list[str], config: dict, output: Path | None) -> dict:
    """Label prepared pool rows. Caller has already required a key and a pool path."""
    from router.isolation import load_exclusions
    from router.prepare_pool import prepare_pool

    prepared = prepare_pool(rows, load_exclusions(root / config["sealed_labels"]))
    model = config["teacher"]["model"]
    if not isinstance(model, str) or not model.strip():
        raise RouterWall("WALL: teacher model is unset")
    temperature = config["teacher"]["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        temperature = 0
    labelled = label_rows(prepared.rows, labels, tinker_teacher(model, float(temperature)))
    floor = confidence_floor(config)
    kept = filter_for_train(labelled, labels, floor)
    dest = output or (root / "data" / "router" / "teacher_labels.jsonl")
    refuse_sealed_input(dest, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
    dest.write_text(payload, encoding="utf-8")
    return {
        "status": "labelled",
        "teacher_model": model,
        "input_rows": prepared.input_rows,
        "empty_text": prepared.empty_text,
        "duplicate_doc_ids": prepared.duplicate_doc_ids,
        "sealed_overlaps": prepared.sealed_overlaps,
        "labelled": len(labelled),
        "kept": len(kept),
        "excluded": len(labelled) - len(kept),
        "wrote": True,
        "output": dest.name,
    }


def tinker_teacher(model: str, temperature: float):
    """One completion from the fixed teacher. The API key is never printed."""
    import tinker
    from tinker import types

    service = tinker.ServiceClient()
    client = service.create_sampling_client(base_model=model)
    tokenizer = client.get_tokenizer()
    params = types.SamplingParams(max_tokens=128, temperature=temperature, top_p=1, seed=0)

    def respond(messages: list[dict[str, str]]) -> str:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer.encode(text)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        result = client.sample(
            prompt=types.ModelInput.from_ints([int(token) for token in encoded]),
            num_samples=1,
            sampling_params=params,
        ).result()
        try:
            return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(result.sequences[0].tokens)

    return respond


def _json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[A-Za-z0-9]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    return obj


def _confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < 0.0 or number > 1.0:
        return None
    return number


if __name__ == "__main__":
    raise SystemExit(main())
