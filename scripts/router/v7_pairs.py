"""Router v7 training pairs, generated from the v9 definitions and the training rows only.

Three generators, all generic. Labels, cue words and windows are read from
``configs/router/label_set_v9.json`` and ``configs/router/pairs_v7.json``. Nothing here names a
label, project, document, row or form number, and no document text is written into code.

Format (fits the single-label constrained-decoding router): a pair row is the same chat as a
training row (system prompt, document text, one-label answer) with a restricted label menu.
The system prompt is ``router.teacher_label.prompt_for(menu)``, so each listed label keeps its
approved definition and Is-not line verbatim. The answer is the row's own teacher label, which
is always in the menu. A contrastive row puts a document beside one neighbour named in the
Is-not lines, so the loss is spent on that one distinction. The parser and the full 22-label
prompt used for validation and sealed scoring are unchanged.

1b contrastive: neighbours are the labels named (by label name or by an unambiguous regional
   name) in each definition's Is-not line, made symmetric. Each class's eligible rows are
   ordered by a seeded hash; the first min(eligible, budget) rows get one neighbour each,
   round-robin over the class's neighbours. budget = median per-class row count of train.
1c combined form: field state from content. certified amount present AND certifier
   signature/date present -> certificate side; certifier cue present, an amount present, no
   certified amount and no certifier signature/date -> application side. Only rows whose field
   state agrees with their teacher label qualify. A pair is one application-side and one
   certificate-side document from the same family; unmatched documents make no pair.
1d catch-all: rows of a catch-all label that pass the content gate and fail every
   instrument test (no instrument label name or regional name in the head of the text). Menu =
   own catch-all label + every instrument label. At most ``budget`` rows per catch-all label.

Pair files store references (doc_id, file_sha256, label, menu), not text. ``materialise``
rebuilds the chats from data/router/train.jsonl and refuses any mismatch.

v7b options (``configs/router/pairs_v7b.json``; absent keys keep the v7 behaviour above byte for byte):
row_order = project_round_robin: within a class, projects are ordered by a seeded hash and their
   rows (each in seeded-hash order) are interleaved, so one project cannot fill a class budget.
contrastive.neighbour_choice = named_in_head_first: rows whose own text head (first head_chars)
   names one of the class's Is-not neighbours (label name or unambiguous regional name) fill the
   budget first; such a row is paired with the least-used named neighbour whose edge is below
   edge_cap_factor x the round-robin share (ceil(rows / neighbours)); otherwise, and for rows
   naming none, with the least-used neighbour (ties in label-set order).
catch_all.sibling_pairs: each selected catch-all row also gets a two-label menu with one sibling
   catch-all label, the least-used sibling (ties in label-set order).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from collect.content_gate import gate
from router.guard import RouterWall
from router.labels import load_label_set, load_labels
from router.teacher_label import prompt_for

PAIR_FILES = {
    "contrastive": "data/router/v7/pairs_contrastive.jsonl",
    "combined_form": "data/router/v7/pairs_combined_form.jsonl",
    "catch_all": "data/router/v7/pairs_catch_all.jsonl",
}
SUMMARY_FILE = "data/router/v7/pairs_summary.json"

_SUBORDINATE = re.compile(r"\b(?:when|if|unless|under|that|framed|called)\b")
_ALT_SPLIT = re.compile(r"\s/\s|\bor\b")
_LETTER = re.compile(r"[A-Za-z]")
_AMOUNT = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d{3,}\.\d{2})(?![\d,])")
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_DATE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b" + _MONTH + r" \d{1,2},? \d{4}\b"
    r"|\b\d{1,2}[ -]" + _MONTH + r"[ -]\d{2,4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)


# ---------------------------------------------------------------- label-set parsing

def _alias_phrases(text: str) -> list[str]:
    """Split a regional-names sentence into phrases: commas/semicolons outside brackets, " / ", "or";
    bracketed asides and trailing qualifier clauses are dropped."""
    parts: list[str] = []
    depth, current = 0, ""
    for char in text:
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if char in ",;":
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    phrases: list[str] = []
    for part in parts:
        for piece in _ALT_SPLIT.split(part):
            piece = _SUBORDINATE.split(piece)[0].strip(" .:\"'").strip()
            if _LETTER.search(piece):
                phrases.append(piece)
    return phrases


def label_aliases(label_set: dict) -> dict[str, str]:
    """casefolded phrase -> label, from each label name and its regional names. A phrase claimed by two labels is dropped."""
    claims: dict[str, set[str]] = defaultdict(set)
    for item in label_set["labels"]:
        label = item["label"]
        claims[label.casefold()].add(label)
        for phrase in _alias_phrases(item.get("regional_names_text") or ""):
            claims[phrase.casefold()].add(label)
    return {phrase: next(iter(owners)) for phrase, owners in claims.items() if len(owners) == 1}


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w/])" + re.escape(phrase) + r"(?:s|es)?(?![\w/])", re.I)


class AliasMatcher:
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = dict(aliases)
        self._patterns = [(phrase, label, _phrase_pattern(phrase)) for phrase, label in sorted(aliases.items())]

    def labels_in(self, text: str, among: set[str] | None = None) -> set[str]:
        found = set()
        for _phrase, label, pattern in self._patterns:
            if among is not None and label not in among:
                continue
            if label not in found and pattern.search(text):
                found.add(label)
        return found


def is_not_neighbours(label_set: dict) -> dict[str, list[str]]:
    """label -> labels named in its own Is-not line (label names and unambiguous regional names), plus the reverse direction."""
    order = [item["label"] for item in label_set["labels"]]
    matcher = AliasMatcher(label_aliases(label_set))
    named: dict[str, set[str]] = {}
    for item in label_set["labels"]:
        named[item["label"]] = matcher.labels_in(item.get("is_not") or "") - {item["label"]}
    symmetric: dict[str, set[str]] = {label: set(named[label]) for label in order}
    for label, others in named.items():
        for other in others:
            symmetric[other].add(label)
    return {label: [x for x in order if x in symmetric[label]] for label in order}


# ---------------------------------------------------------------- field state (1c)

def _nonzero_amounts(text: str) -> list[float]:
    values = []
    for match in _AMOUNT.finditer(text):
        value = float(match.group().replace(",", ""))
        if value > 0:
            values.append(value)
    return values


def _cue_pattern(cues: list[str]) -> re.Pattern[str]:
    parts = sorted((re.escape(cue) for cue in cues), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")[\w']*", re.I)


def field_state(text: str, rules: dict) -> dict:
    """Combined-form field state from content only (see module doc)."""
    window = int(rules["window_chars"])
    value_window = int(rules["value_window_chars"])
    certified = re.compile(r"\b" + re.escape(rules["certified_cue"]) + r"\b", re.I)
    certifier = _cue_pattern(rules["certifier_cues"])
    signature = _cue_pattern(rules["signature_cues"])
    certified_amount = any(_nonzero_amounts(text[m.end(): m.end() + window]) for m in certified.finditer(text))
    certifier_cue = False
    signed_dated = False
    for match in certifier.finditer(text):
        certifier_cue = True
        block = text[match.end(): match.end() + window]
        for sig in signature.finditer(block):
            if _DATE.search(block[sig.end(): sig.end() + value_window]):
                signed_dated = True
                break
        if signed_dated:
            break
    amount = bool(_nonzero_amounts(text))
    if certified_amount and signed_dated:
        state = "certificate"
    elif certifier_cue and amount and not certified_amount and not signed_dated:
        state = "application"
    else:
        state = "indeterminate"
    return {
        "state": state,
        "certified_amount": certified_amount,
        "certifier_signature_date": signed_dated,
        "certifier_cue": certifier_cue,
        "amount": amount,
    }


# ---------------------------------------------------------------- generation

def _row_label(row: dict) -> str:
    return row["messages"][-1]["content"]


def _row_text(row: dict) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def _order_key(seed: int, *parts: str) -> str:
    return hashlib.sha256("|".join([str(seed), *parts]).encode("utf-8")).hexdigest()


def pair_files(rules: dict) -> dict[str, str]:
    """Pair file per generator under the rules' output_dir (v7 default: data/router/v7)."""
    out = rules.get("output_dir", "data/router/v7")
    keys = ["contrastive", "combined_form", "catch_all"]
    if rules["catch_all"].get("sibling_pairs"):
        keys.append("catch_all_sibling")
    return {key: f"{out}/pairs_{key}.jsonl" for key in keys}


def summary_file(rules: dict) -> str:
    return f'{rules.get("output_dir", "data/router/v7")}/pairs_summary.json'


def _project_round_robin(rows: list[dict], seed: int, label: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("project_id"))].append(row)
    projects = sorted(groups, key=lambda project: _order_key(seed, label, "project", project))
    out: list[dict] = []
    depth = max((len(g) for g in groups.values()), default=0)
    for i in range(depth):
        for project in projects:
            if i < len(groups[project]):
                out.append(groups[project][i])
    return out


def _least_used(candidates: list[str], used: Counter, label: str) -> str:
    return min(candidates, key=lambda n: (used[f"{label} | {n}"], candidates.index(n)))


def class_budget(train_rows: list[dict]) -> int:
    counts = Counter(_row_label(row) for row in train_rows)
    return int(statistics.median(counts.values()))


def _ref(row: dict, menu: list[str], pair_type: str, **extra) -> dict:
    ref = {
        "pair_type": pair_type,
        "doc_id": row["doc_id"],
        "file_sha256": row.get("file_sha256"),
        "project_id": row.get("project_id"),
        "label": _row_label(row),
        "menu": menu,
    }
    ref.update(extra)
    return ref


def build_pairs(train_rows: list[dict], label_set: dict, labels: list[str], rules: dict) -> tuple[dict[str, list[dict]], dict]:
    order = [item["label"] for item in label_set["labels"] if item["label"] in set(labels)]
    rank = {label: i for i, label in enumerate(order)}
    seed = int(rules["selection_seed"])
    budget = class_budget(train_rows)
    menu_of = lambda chosen: sorted(set(chosen), key=rank.__getitem__)  # noqa: E731
    by_label: dict[str, list[dict]] = defaultdict(list)
    gate_fail = Counter()
    for row in train_rows:
        if not isinstance(row.get("doc_id"), str):
            raise RouterWall("WALL: a training row has no doc_id")
        if gate(_row_text(row))[0]:
            by_label[_row_label(row)].append(row)
        else:
            gate_fail[_row_label(row)] += 1
    row_order = rules.get("row_order", "hash")
    if row_order not in ("hash", "project_round_robin"):
        raise RouterWall("WALL: unknown row_order")
    for label in by_label:
        by_label[label].sort(key=lambda row: _order_key(seed, label, row["doc_id"]))
        if row_order == "project_round_robin":
            by_label[label] = _project_round_robin(by_label[label], seed, label)
    matcher = AliasMatcher(label_aliases(label_set))

    # 1c combined form
    cf = rules["combined_form"]
    app_label, cert_label = cf["labels"]["application"], cf["labels"]["certificate"]
    states: dict[str, Counter] = {app_label: Counter(), cert_label: Counter()}
    sides: dict[str, dict[str, list[dict]]] = {"application": defaultdict(list), "certificate": defaultdict(list)}
    for label, side in ((app_label, "application"), (cert_label, "certificate")):
        for row in by_label.get(label, []):
            state = field_state(_row_text(row), cf)["state"]
            states[label][state] += 1
            if state == side:
                sides[side][str(row.get(cf["family"]))].append(row)
    combined: list[dict] = []
    families_paired = []
    for family in sorted(set(sides["application"]) & set(sides["certificate"])):
        apps, certs = sides["application"][family], sides["certificate"][family]
        n = min(len(apps), len(certs))
        families_paired.append({"family_rows": [len(apps), len(certs)], "pairs": n})
        for k in range(n):
            pair_id = _order_key(seed, "combined", apps[k]["doc_id"], certs[k]["doc_id"])[:16]
            for row, side in ((apps[k], "application"), (certs[k], "certificate")):
                combined.append(_ref(row, menu_of([app_label, cert_label]), "combined_form", field_state=side, pair_id=pair_id))

    # 1b contrastive
    neighbours = is_not_neighbours(label_set)
    governed = {app_label, cert_label}
    contrastive: list[dict] = []
    edge_counts: Counter = Counter()
    choice = rules["contrastive"].get("neighbour_choice", "round_robin")
    if choice not in ("round_robin", "named_in_head_first"):
        raise RouterWall("WALL: unknown contrastive.neighbour_choice")
    named_rows: Counter = Counter()
    for label in order:
        hood = [n for n in neighbours.get(label, []) if n in rank and not ({label, n} == governed)]
        if not hood:
            continue
        if choice == "round_robin":
            for i, row in enumerate(by_label.get(label, [])[:budget]):
                neighbour = hood[i % len(hood)]
                contrastive.append(_ref(row, menu_of([label, neighbour]), "contrastive", neighbour=neighbour))
                edge_counts[f"{label} | {neighbour}"] += 1
            continue
        head_c = int(rules["contrastive"]["head_chars"])
        named = [(row, [n for n in hood if n in matcher.labels_in(_row_text(row)[:head_c], set(hood))]) for row in by_label.get(label, [])]
        ordered = ([x for x in named if x[1]] + [x for x in named if not x[1]])[:budget]
        edge_cap = float(rules["contrastive"]["edge_cap_factor"]) * -(-len(ordered) // len(hood))
        for row, in_head in ordered:
            open_named = [n for n in in_head if edge_counts[f"{label} | {n}"] < edge_cap]
            neighbour = _least_used(open_named or hood, edge_counts, label)
            if open_named:
                named_rows[label] += 1
            contrastive.append(_ref(row, menu_of([label, neighbour]), "contrastive", neighbour=neighbour, named_in_head=bool(open_named)))
            edge_counts[f"{label} | {neighbour}"] += 1

    # 1d catch-all
    ca = rules["catch_all"]
    catch_all = [label for label in ca["labels"] if label in rank]
    instruments = [label for label in order if label not in set(catch_all)]
    head = int(ca["head_chars"])
    catch_rows: list[dict] = []
    sibling_rows: list[dict] = []
    sibling_edges: Counter = Counter()
    qualifying: dict[str, int] = {}
    for label in catch_all:
        passed = [row for row in by_label.get(label, []) if not matcher.labels_in(_row_text(row)[:head], set(instruments))]
        qualifying[label] = len(passed)
        siblings = [x for x in catch_all if x != label]
        for row in passed[:budget]:
            catch_rows.append(_ref(row, menu_of([label, *instruments]), "catch_all"))
            if ca.get("sibling_pairs") and siblings:
                sibling = _least_used(siblings, sibling_edges, label)
                sibling_rows.append(_ref(row, menu_of([label, sibling]), "catch_all_sibling", neighbour=sibling))
                sibling_edges[f"{label} | {sibling}"] += 1

    pairs = {"contrastive": contrastive, "combined_form": combined, "catch_all": catch_rows}
    if ca.get("sibling_pairs"):
        pairs["catch_all_sibling"] = sibling_rows
    summary = {
        "train_rows": len(train_rows),
        "content_gate_failed_by_label": dict(sorted(gate_fail.items())),
        "per_class_budget": budget,
        "neighbours": neighbours,
        "contrastive": {"rows": len(contrastive), "by_label": dict(Counter(r["label"] for r in contrastive)), "by_edge": dict(sorted(edge_counts.items()))},
        "combined_form": {
            "field_state_by_teacher_label": {label: dict(sorted(c.items())) for label, c in states.items()},
            "qualifying": {side: sum(len(v) for v in sides[side].values()) for side in sides},
            "families_with_both_sides": families_paired,
            "rows": len(combined),
        },
        "catch_all": {"instrument_labels": instruments, "qualifying": qualifying, "rows": len(catch_rows), "by_label": dict(Counter(r["label"] for r in catch_rows))},
    }
    if row_order != "hash":
        summary["row_order"] = row_order
    if choice != "round_robin":
        summary["contrastive"]["named_in_head_rows_by_label"] = dict(named_rows)
    if ca.get("sibling_pairs"):
        summary["catch_all_sibling"] = {
            "rows": len(sibling_rows),
            "by_edge": dict(sorted(sibling_edges.items())),
        }
    return pairs, summary


def materialise(refs: list[dict], train_rows: list[dict], label_set: dict) -> list[dict]:
    """Chats for pair refs, rebuilt from the training rows. Refuses a ref whose row or label does not match."""
    by_id = {row["doc_id"]: row for row in train_rows}
    out = []
    for ref in refs:
        row = by_id.get(ref["doc_id"])
        if row is None or _row_label(row) != ref["label"] or row.get("file_sha256") != ref.get("file_sha256"):
            raise RouterWall("WALL: a pair ref does not match a training row")
        if ref["label"] not in ref["menu"] or len(ref["menu"]) < 2:
            raise RouterWall("WALL: a pair menu does not hold its label")
        item = {
            "project_id": row.get("project_id"),
            "doc_id": row["doc_id"],
            "pair_type": ref["pair_type"],
            "messages": [
                {"role": "system", "content": prompt_for(ref["menu"], label_set=label_set)},
                {"role": "user", "content": _row_text(row)},
                {"role": "assistant", "content": ref["label"]},
            ],
        }
        if isinstance(row.get("file_sha256"), str):
            item["file_sha256"] = row["file_sha256"]
        out.append(item)
    return out


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


RULES_FILE = "configs/router/pairs_v7.json"


def load_rules(root: Path, rules_file: str = RULES_FILE) -> dict:
    return json.loads((root / rules_file).read_text(encoding="utf-8"))


def generate(root: Path, rules_file: str = RULES_FILE) -> tuple[dict[str, list[dict]], dict]:
    rules = load_rules(root, rules_file)
    label_set = load_label_set(root / rules["label_set"])
    labels = load_labels(root / "configs" / "router" / "labels.json")
    train_rows = load_rows(root / "data" / "router" / "train.jsonl")
    return build_pairs(train_rows, label_set, labels, rules)


def v7_training_examples(root: Path, rules_file: str = RULES_FILE) -> tuple[list[dict], list[dict]]:
    """(examples, pair_refs): the merged train rows unchanged, then every committed pair ref materialised in file order."""
    from router.train_router import require_no_paths

    rules = load_rules(root, rules_file)
    label_set = load_label_set(root / rules["label_set"])
    base = load_rows(root / "data" / "router" / "train.jsonl")
    refs: list[dict] = []
    for rel in pair_files(rules).values():
        refs.extend(load_rows(root / rel))
    return require_no_paths(base + materialise(refs, base, label_set)), refs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate router v7 pair refs and the tempered class weights")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="Regenerate in memory and compare with the committed files")
    parser.add_argument("--train-config", default=None, help="v7-family train config (default configs/router/train_v7.json)")
    args = parser.parse_args(argv)
    from router.balance_v7 import CONFIG_FILE, load_v7_config, weights_payload, v7_example_labels

    root = args.root.resolve()
    config_file = args.train_config or CONFIG_FILE
    cfg = load_v7_config(root, config_file)
    rules_file = cfg["config"]["training_examples"]["pair_rules"]
    rules = load_rules(root, rules_file)
    paths = pair_files(rules)
    if list(paths.values()) != cfg["config"]["training_examples"]["pairs"]:
        raise RouterWall("WALL: train config pair files do not match the pair rules")
    pairs, summary = generate(root, rules_file)
    files = {paths[k]: "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in v) for k, v in pairs.items()}
    files[summary_file(rules)] = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    base = load_rows(root / "data" / "router" / "train.jsonl")
    labels_all = v7_example_labels(base, [r for v in pairs.values() for r in v])
    payload = weights_payload(root, labels_all, config_file)
    files[cfg["weights_file"]] = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    run = cfg["config"]["run"]
    if args.check:
        stale = [p for p, text in files.items() if not (root / p).is_file() or (root / p).read_text(encoding="utf-8") != text]
        if stale:
            print("stale: " + ", ".join(stale))
            return 1
        print(f"{run} pairs and weights match the committed files")
        return 0
    for p, text in files.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(text, encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in pairs.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
