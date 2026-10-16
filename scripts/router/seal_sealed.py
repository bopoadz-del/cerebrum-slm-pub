"""Seal practitioner-labelled router sheets.

The ticked column is the owner's label. It is copied, not edited, and not
inferred from a path, filename, folder, id, or a meta file. Unsure rows and
rows marked excluded_from_eval are omitted. A hidden proposal file is used
only for an agreement statistic. A meta file may add the content sha256,
isolation key, and source URL. It never supplies a label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from router.guard import RouterWall
from router.labels import load_labels, load_sealed_config, repo_root
from router.sheet import UNSURE

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_TICK = {"1", "x", "y", "yes", "true", "t", "✓", "✔"}
_FALSE = {"", "0", "n", "no", "false", "f"}


def corpus_project(source_path: object) -> str | None:
    """Top-level corpus folder. A bare filename is not a project and not a label."""
    if not isinstance(source_path, str):
        return None
    text = source_path.strip().replace("\\", "/")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if len(parts) < 2:
        return None
    return parts[0]


MAX_PER_PROJECT_PER_CLASS = 3


def project_key(row: dict) -> str:
    """Source project for the diversity rule. Counts projects, not files.

    ``web_public`` rows carry a ``source_url``; the project is the host plus the
    first path segment, so every document from one public body counts as one
    project regardless of how many files it published.
    """
    if row.get("project") == "web_public":
        url = row.get("source_url") or ""
        parts = url.replace("https://", "").replace("http://", "").split("/")
        host = parts[0] if parts else ""
        first = parts[1] if len(parts) > 1 else ""
        return f"{host}/{first}" if first else host
    return str(row.get("project") or "")


def enforce_project_diversity(sealed: list[dict], limit: int = MAX_PER_PROJECT_PER_CLASS) -> None:
    """At most ``limit`` sealed rows per (source project, label). Raises on breach."""
    counts: dict[tuple[str, str], int] = {}
    for row in sealed:
        key = (project_key(row), str(row.get("label") or ""))
        counts[key] = counts.get(key, 0) + 1
    for (project, label), count in counts.items():
        if count > limit:
            raise RouterWall(
                f"WALL: {count} sealed rows for {label!r} from project {project!r} "
                f"(limit {limit})"
            )


def seal_sheet(
    sheet: Path | list[Path],
    output_dir: Path,
    labels: list[str],
    *,
    proposed: Path | None = None,
    meta: list[Path] | None = None,
) -> dict:
    """Validate the sheets and write sealed labels. Raises without writing on a bad tick."""
    paths = [sheet] if isinstance(sheet, Path) else list(sheet)
    if not paths:
        raise RouterWall("WALL: no sheet")
    rows = _collect_rows(paths, labels)
    sealed, dropped = _select(rows, labels, _read_meta(list(meta or [])))
    enforce_project_diversity(sealed)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "sealed_labels.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sealed)
    labels_path.write_text(payload, encoding="utf-8", newline="\n")
    written = ["sealed_labels.jsonl"]
    agreement_path = output_dir / "agreement.json"
    if proposed is not None:
        before = labels_path.read_bytes()
        stats = agreement(sealed, _read_proposed(proposed), labels)
        agreement_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        if labels_path.read_bytes() != before:
            raise RouterWall("WALL: agreement rewrote a sealed label")
        written.append("agreement.json")
    _write_digests(output_dir, written)
    return {"sealed": len(sealed), "dropped": dropped, "files": written}


def agreement(sealed: list[dict], proposed: dict[str, str], labels: list[str]) -> dict:
    """Per-label and overall agreement. Does not return replacement labels."""
    per_label: dict[str, float | None] = {}
    compared = 0
    matched = 0
    for label in labels:
        owned = [row for row in sealed if row.get("label") == label and row.get("sheet_id") in proposed]
        if not owned:
            per_label[label] = None
            continue
        hits = sum(1 for row in owned if proposed[row["sheet_id"]] == label)
        per_label[label] = hits / len(owned)
        compared += len(owned)
        matched += hits
    return {
        "compared": compared,
        "overall": None if compared == 0 else matched / compared,
        "per_label": per_label,
        "note": "Agreement does not change sealed labels.",
    }


def read_table(path: Path, labels: list[str] | None = None) -> tuple[list[str], list[dict[str, str]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".xlsx":
        return _read_xlsx(path, labels or [])
    raise RouterWall("WALL: sheet must be csv or xlsx")


def _require_header(header: list[str], labels: list[str]) -> None:
    present = set(header)
    if any(label not in present for label in (*labels, UNSURE)):
        raise RouterWall("WALL: sheet header is missing a declared label column")


def _collect_rows(paths: list[Path], labels: list[str]) -> list[dict[str, str]]:
    """Read every sheet. ``id`` is the sheet id when ``sheet_id`` is absent."""
    collected: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        header, rows = read_table(path, labels)
        _require_header(header, labels)
        for index, row in enumerate(rows, start=2):
            sheet_id = _resolve_sheet_id(row, index)
            if sheet_id in seen:
                raise RouterWall(f"WALL: duplicate sheet id {sheet_id}")
            seen.add(sheet_id)
            copied = dict(row)
            copied["sheet_id"] = sheet_id
            collected.append(copied)
    return collected


def _resolve_sheet_id(row: dict[str, str], index: int) -> str:
    sheet_id = _blank_none(row.get("sheet_id"))
    alias = _blank_none(row.get("id"))
    if sheet_id and alias and sheet_id != alias:
        raise RouterWall(f"WALL: {sheet_id} id column does not match sheet_id")
    if sheet_id:
        return sheet_id
    if alias:
        return alias
    return f"row-{index}"


def _select(
    rows: list[dict[str, str]],
    labels: list[str],
    meta_by_id: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    allowed = [*labels, UNSURE]
    joined = meta_by_id or {}
    sealed: list[dict] = []
    dropped = {"unsure": 0, "excluded_from_eval": 0}
    for index, row in enumerate(rows, start=2):
        ticked = _ticked(row, allowed)
        sheet_id = _resolve_sheet_id(row, index)
        if len(ticked) != 1:
            raise RouterWall(f"WALL: {sheet_id} does not have exactly one tick")
        label = ticked[0]
        if _excluded(row.get("excluded_from_eval")):
            dropped["excluded_from_eval"] += 1
            continue
        if label == UNSURE:
            dropped["unsure"] += 1
            continue
        sealed.append(_sealed_row(row, label, sheet_id, joined.get(sheet_id)))
    return sealed, dropped


def _sealed_row(
    row: dict[str, str],
    label: str,
    sheet_id: str,
    meta_row: dict[str, str] | None = None,
) -> dict:
    """Copy the owner tick. Path fields and meta labels stay out of the label."""
    sealed = {
        "sheet_id": sheet_id,
        "label": label,
        "file_sha256": _blank_none(row.get("file_sha256")),
        "project": corpus_project(row.get("source_path")),
    }
    if meta_row is not None:
        content_sha = meta_row.get("file_sha256")
        if content_sha:
            sealed["file_sha256"] = content_sha
        sealed["project"] = "web_public"
        isolation_key = meta_row.get("isolation_key")
        if isolation_key:
            sealed["isolation_key"] = isolation_key
        source_url = meta_row.get("source_url")
        if source_url:
            sealed["source_url"] = source_url
    excerpt = row.get("excerpt")
    if isinstance(excerpt, str) and excerpt.strip():
        sealed["excerpt"] = excerpt
    return sealed


def _ticked(row: dict[str, str], columns: list[str]) -> list[str]:
    ticked: list[str] = []
    for column in columns:
        if _is_tick(row.get(column)):
            ticked.append(column)
    return ticked


def _is_tick(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    folded = text.casefold()
    if folded in _FALSE:
        return False
    return folded in _TICK or text in _TICK


def _excluded(value: object) -> bool:
    return _is_tick(value)


def _blank_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RouterWall("WALL: sheet has no header")
        header = [name for name in reader.fieldnames if name]
        rows = [{key: (value or "") for key, value in row.items() if key is not None} for row in reader]
    return header, rows


def _read_xlsx(path: Path, labels: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise RouterWall("WALL: xlsx is not a zip") from exc
    with archive:
        shared = _shared_strings(archive)
        sheet = _chosen_sheet(archive, shared, labels)
        grid = _sheet_rows(sheet, shared)
    if not grid:
        raise RouterWall("WALL: sheet has no header")
    header = [cell.strip() for cell in grid[0]]
    rows: list[dict[str, str]] = []
    width = len(header)
    for raw in grid[1:]:
        if not any(cell.strip() for cell in raw):
            continue
        padded = raw + [""] * (width - len(raw))
        rows.append({header[index]: padded[index] for index in range(width) if header[index]})
    return header, rows


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("m:si", _NS):
        strings.append("".join(node.text or "" for node in item.iterfind(".//m:t", _NS)))
    return strings


_LABELLING_SHEET = "Labelling"
_RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_REL_NS = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _first_sheet(archive: zipfile.ZipFile) -> bytes:
    names = [name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")]
    if not names:
        raise RouterWall("WALL: xlsx has no worksheet")
    return archive.read(sorted(names)[0])


def _chosen_sheet(archive: zipfile.ZipFile, shared: list[str], labels: list[str]) -> bytes:
    """Prefer the sheet named Labelling. Otherwise the first sheet whose header has every declared label."""
    ordered = _workbook_sheets(archive)
    if not ordered:
        return _first_sheet(archive)
    for name, path in ordered:
        if name == _LABELLING_SHEET:
            return archive.read(path)
    for _name, path in ordered:
        payload = archive.read(path)
        if _header_has_labels(payload, shared, labels):
            return payload
    raise RouterWall("WALL: sheet header is missing a declared label column")


def _workbook_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    if "xl/workbook.xml" not in archive.namelist():
        return []
    targets: dict[str, str] = {}
    rels_name = "xl/_rels/workbook.xml.rels"
    if rels_name in archive.namelist():
        rels = ElementTree.fromstring(archive.read(rels_name))
        for rel in rels.findall("p:Relationship", _REL_NS):
            rid = rel.get("Id")
            target = rel.get("Target") or ""
            if rid and target:
                targets[rid] = _worksheet_target(target)
    root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    ordered: list[tuple[str, str]] = []
    for sheet in root.findall("m:sheets/m:sheet", _NS):
        name = (sheet.get("name") or "").strip()
        path = targets.get(sheet.get(_RID) or "")
        if path and path in archive.namelist():
            ordered.append((name, path))
    return ordered


def _worksheet_target(target: str) -> str:
    cleaned = target.lstrip("/")
    if cleaned.startswith("xl/"):
        return cleaned
    return "xl/" + cleaned


def _header_has_labels(payload: bytes, shared: list[str], labels: list[str]) -> bool:
    grid = _sheet_rows(payload, shared)
    if not grid or not labels:
        return False
    header = {cell.strip() for cell in grid[0]}
    return all(label in header for label in labels)


def _sheet_rows(payload: bytes, shared: list[str]) -> list[list[str]]:
    root = ElementTree.fromstring(payload)
    rows: list[list[str]] = []
    for row in root.findall("m:sheetData/m:row", _NS):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", _NS):
            ref = cell.get("r") or ""
            column = _column_index(ref)
            cells[column] = _cell_text(cell, shared)
        if not cells:
            continue
        width = max(cells) + 1
        rows.append([cells.get(index, "") for index in range(width)])
    return rows


def _column_index(ref: str) -> int:
    letters = "".join(char for char in ref if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - 64)
    return max(index - 1, 0)


def _cell_text(cell: ElementTree.Element, shared: list[str]) -> str:
    kind = cell.get("t")
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.iterfind(".//m:t", _NS))
    value = cell.find("m:v", _NS)
    if value is None or value.text is None:
        return ""
    if kind == "s":
        try:
            return shared[int(value.text)]
        except (ValueError, IndexError):
            return ""
    return value.text


def _read_meta(paths: list[Path]) -> dict[str, dict[str, str]]:
    """Join fields by sheet id. ``proposed_label`` and document text are ignored."""
    joined: dict[str, dict[str, str]] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RouterWall(f"WALL: meta file is unreadable: {path.name}") from exc
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RouterWall(f"WALL: {path.name}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RouterWall(f"WALL: {path.name}:{lineno}: record is not an object")
            sheet_id = _meta_id(obj, path.name, lineno)
            if sheet_id in joined:
                raise RouterWall(f"WALL: duplicate meta id {sheet_id}")
            record: dict[str, str] = {}
            for source_key, dest_key in (
                ("content_sha256", "file_sha256"),
                ("isolation_key", "isolation_key"),
                ("source_url", "source_url"),
            ):
                value = obj.get(source_key)
                if isinstance(value, str) and value.strip():
                    record[dest_key] = value.strip()
            joined[sheet_id] = record
    return joined


def _meta_id(obj: dict, name: str, lineno: int) -> str:
    sheet_id = obj.get("sheet_id")
    alias = obj.get("id")
    sheet_text = sheet_id.strip() if isinstance(sheet_id, str) else ""
    alias_text = alias.strip() if isinstance(alias, str) else ""
    if sheet_text and alias_text and sheet_text != alias_text:
        raise RouterWall(f"WALL: {name}:{lineno}: id does not match sheet_id")
    chosen = sheet_text or alias_text
    if not chosen:
        raise RouterWall(f"WALL: {name}:{lineno}: meta row has no sheet_id")
    return chosen


def _read_proposed(path: Path) -> dict[str, str]:
    proposed: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RouterWall(f"WALL: {path}:{lineno}: {exc}") from exc
        if not isinstance(obj, dict):
            raise RouterWall(f"WALL: {path}:{lineno}: record is not an object")
        sheet_id = obj.get("sheet_id")
        label = obj.get("proposed_label")
        if isinstance(sheet_id, str) and isinstance(label, str):
            proposed[sheet_id] = label
    return proposed


def _write_digests(directory: Path, names: list[str]) -> None:
    lines: list[str] = []
    for name in names:
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        line = f"{digest}  {name}\n"
        lines.append(line)
        (directory / f"{name}.sha256").write_text(line, encoding="utf-8", newline="\n")
    (directory / "SHA256SUMS").write_text("".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal practitioner router labelling sheets")
    parser.add_argument("--sheet", type=Path, action="append", required=True)
    parser.add_argument("--meta", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--proposed", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.root or repo_root()).resolve()
    try:
        labels = load_labels(root / "configs" / "router" / "labels.json")
        config = load_sealed_config(root / "configs" / "router" / "sealed_eval.json")
        output_dir = args.output_dir or (root / config["sealed_dir"])
        report = seal_sheet(
            args.sheet,
            output_dir,
            labels,
            proposed=args.proposed,
            meta=args.meta,
        )
    except RouterWall as exc:
        print(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
