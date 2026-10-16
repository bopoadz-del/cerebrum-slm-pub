"""NO HARDWIRING: probe ids, domain names, and word lists stay out of src/."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from gates.catalog import Catalog, CatalogError, build_catalog
from gates.findings import Finding
from gates.records import (
    RecordParseError,
    content_hash,
    iter_training_files,
    load_records,
    looks_like_pair,
    pair_view,
    record_hash,
    relpath,
)

_WORD_LIST_NAME = re.compile(
    r"(?i)^(word_?lists?|lexicons?|vocabs?|vocabulary|keywords?|probe_?ids?|probes|"
    r"domain_?names?|domains|brands?|site_?names?|sites|block_?lists?|deny_?lists?|"
    r"stoplists?)$"
)
_NAMED_MIN = 3
_UNNAMED_MIN = 8
_SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pyc",
    ".so",
    ".whl",
    ".parquet",
    ".zip",
    ".gz",
}


def check_hardwiring(root: Path) -> list[Finding]:
    try:
        catalog = build_catalog(root)
    except CatalogError as exc:
        return [Finding("no-hardwiring", exc.path, str(exc))]
    findings = scan_src(root, catalog)
    findings.extend(scan_training_hashes(root, catalog))
    return findings


def scan_src(root: Path, catalog: Catalog) -> list[Finding]:
    src = root / "src"
    if not src.is_dir():
        return []
    token_re = _token_pattern(catalog.probe_ids | catalog.domains)
    findings: list[Finding] = []
    for path in sorted(item for item in src.rglob("*") if item.is_file()):
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        where = relpath(path, root)
        if token_re is not None:
            for match in token_re.finditer(text):
                token = match.group(0)
                kind = "probe id" if _known(token, catalog.probe_ids) else "domain name"
                findings.append(
                    Finding(
                        "no-hardwiring",
                        where,
                        f"{kind} {token!r} is hardcoded in src/",
                    )
                )
        findings.extend(_word_lists(path, text, where))
    return findings


def scan_training_hashes(root: Path, catalog: Catalog) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_training_files(root):
        where = relpath(path, root)
        try:
            rows = load_records(path)
        except RecordParseError as exc:
            findings.append(Finding("no-hardwiring", where, str(exc)))
            continue
        for lineno, obj in rows:
            if not looks_like_pair(obj):
                continue
            if record_hash(obj) in catalog.record_hashes:
                findings.append(
                    Finding(
                        "no-hardwiring",
                        where,
                        f"line {lineno}: training pair record hash matches a sealed eval item",
                    )
                )
                continue
            view = pair_view(obj)
            if view is None:
                continue
            if content_hash(view.user, view.assistant) in catalog.content_hashes:
                findings.append(
                    Finding(
                        "no-hardwiring",
                        where,
                        f"line {lineno}: training pair content hash matches a sealed eval item",
                    )
                )
    return findings


def _known(token: str, tokens: frozenset[str]) -> bool:
    folded = token.casefold()
    return any(item.casefold() == folded for item in tokens)


def _token_pattern(tokens: frozenset[str]) -> re.Pattern[str] | None:
    if not tokens:
        return None
    parts = sorted((re.escape(token) for token in tokens), key=len, reverse=True)
    # Case-sensitive: a comment that does not repeat the sealed token stays green.
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b")


def _string_consts(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        elements = list(node.elts)
    elif isinstance(node, ast.Dict):
        elements = [key for key in node.keys if key is not None]
    else:
        return []
    return [
        element.value
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]


def _assigned_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    return None


def _word_lists(path: Path, text: str, where: str) -> list[Finding]:
    findings: list[Finding] = []
    suffix = path.suffix.lower()
    if suffix == ".py":
        findings.extend(_python_word_lists(text, where))
    elif suffix == ".json":
        findings.extend(_json_word_lists(path, text, where))
    elif suffix in {".yaml", ".yml"}:
        findings.extend(_yaml_word_lists(path, text, where))
    return findings


def _assigns_dunder_all(targets: list[ast.AST]) -> bool:
    for target in targets:
        if isinstance(target, ast.Name) and target.id == "__all__":
            return True
        if isinstance(target, (ast.List, ast.Tuple)):
            if _assigns_dunder_all(list(target.elts)):
                return True
    return False


def _is_identifier_literal(node: ast.AST) -> bool:
    """True for a list, tuple, or set of Python identifiers, as in __all__."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)) or not node.elts:
        return False
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return False
        if not element.value.isidentifier():
            return False
    return True


def _python_word_lists(text: str, where: str) -> list[Finding]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    findings: list[Finding] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._export_ids: set[int] = set()

        def visit_Assign(self, node: ast.Assign) -> None:
            self._named(node.targets, node.value)
            self._remember_export(node.targets, node.value)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self._named([node.target], node.value)
                self._remember_export([node.target], node.value)
            self.generic_visit(node)

        def visit_List(self, node: ast.List) -> None:
            self._unnamed(node)
            self.generic_visit(node)

        def visit_Tuple(self, node: ast.Tuple) -> None:
            self._unnamed(node)
            self.generic_visit(node)

        def visit_Set(self, node: ast.Set) -> None:
            self._unnamed(node)
            self.generic_visit(node)

        def visit_Dict(self, node: ast.Dict) -> None:
            self._unnamed(node)
            self.generic_visit(node)

        def _remember_export(self, targets: list[ast.AST], value: ast.AST) -> None:
            if _assigns_dunder_all(targets) and _is_identifier_literal(value):
                self._export_ids.add(id(value))

        def _named(self, targets: list[ast.AST], value: ast.AST) -> None:
            strings = _string_consts(value)
            if len(strings) < _NAMED_MIN:
                return
            for target in targets:
                name = _assigned_name(target)
                if name is not None and _WORD_LIST_NAME.fullmatch(name):
                    findings.append(
                        Finding(
                            "no-hardwiring",
                            where,
                            f"word list assigned to {name} ({len(strings)} strings)",
                        )
                    )

        def _unnamed(self, node: ast.AST) -> None:
            if id(node) in self._export_ids:
                return
            strings = _string_consts(node)
            if len(strings) >= _UNNAMED_MIN:
                findings.append(
                    Finding(
                        "no-hardwiring",
                        where,
                        f"word list literal with {len(strings)} strings",
                    )
                )

    Visitor().visit(tree)
    return findings


def _json_word_lists(path: Path, text: str, where: str) -> list[Finding]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    findings: list[Finding] = []
    if isinstance(data, list) and _string_list(data):
        minimum = _NAMED_MIN if _WORD_LIST_NAME.fullmatch(path.stem) else _UNNAMED_MIN
        if len(data) >= minimum:
            findings.append(
                Finding(
                    "no-hardwiring",
                    where,
                    f"word list literal with {len(data)} strings",
                )
            )
    if isinstance(data, dict):
        for key, value in data.items():
            if (
                isinstance(key, str)
                and _WORD_LIST_NAME.fullmatch(key)
                and isinstance(value, list)
                and _string_list(value)
                and len(value) >= _NAMED_MIN
            ):
                findings.append(
                    Finding(
                        "no-hardwiring",
                        where,
                        f"word list assigned to {key} ({len(value)} strings)",
                    )
                )
    return findings


def _yaml_word_lists(path: Path, text: str, where: str) -> list[Finding]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"""-\s+['"]?(.+?)['"]?\s*""", stripped)
        if match is None:
            return []
        items.append(match.group(1))
    minimum = _NAMED_MIN if _WORD_LIST_NAME.fullmatch(path.stem) else _UNNAMED_MIN
    if len(items) >= minimum:
        return [
            Finding(
                "no-hardwiring",
                where,
                f"word list literal with {len(items)} strings",
            )
        ]
    return []


def _string_list(values: list[object]) -> bool:
    return all(isinstance(item, str) for item in values)
