"""Blind-set seal: hash data/eval/** and keep the training path from reading it."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from gates.findings import Finding
from gates.records import RAW_ZIP, iter_training_files, relpath, sha256_file
from private_data import locate, missing_message

_DIGEST = re.compile(r"^[0-9a-fA-F]{64}$")

_READS = (
    re.compile(r"data[/\\]+eval\b"),
    re.compile(r"""["']data["']\s*,\s*["']eval["']"""),
    re.compile(r"""["']data["']\s*\)?\s*/\s*["']eval["']"""),
)


class BlindSealError(Exception):
    """Raised when training code attempts to read data/eval/**."""

    def __init__(self, path: str) -> None:
        super().__init__(f"training path cannot read sealed eval file: {path}")
        self.path = path


@dataclass(frozen=True)
class SealedFile:
    relpath: str
    digest: str


def seal_manifest(root: Path) -> list[SealedFile]:
    base = root / "data" / "eval"
    if not base.is_dir():
        return []
    sealed: list[SealedFile] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        sealed.append(SealedFile(relpath(path, root), sha256_file(path)))
    return sealed


def training_manifest(root: Path) -> list[SealedFile]:
    return [
        SealedFile(relpath(path, root), sha256_file(path))
        for path in iter_training_files(root)
    ]


def training_code_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for folder in (root / "src", root / "scripts" / "train"):
        if folder.is_dir():
            files.extend(path for path in folder.rglob("*") if path.is_file())
    files.extend(path for path in root.glob("train*.py") if path.is_file())
    return sorted(set(files))


def _resolved(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (root.resolve() / path).resolve()


def path_is_sealed(path: Path, root: Path) -> bool:
    eval_root = (root.resolve() / "data" / "eval").resolve()
    try:
        _resolved(path, root).relative_to(eval_root)
    except ValueError:
        return False
    return True


def training_open(path: str | Path, root: Path, mode: str = "rb"):
    """The only training-path opener. It rejects data/eval/** before any read."""
    target = Path(path)
    if path_is_sealed(target, root):
        raise BlindSealError(target.as_posix())
    return _resolved(target, root).open(mode)


def opener_rejects_eval(root: Path) -> bool:
    probe = root / "data" / "eval" / "blind" / ".read-probe"
    try:
        handle = training_open(probe, root)
    except BlindSealError:
        return True
    else:
        handle.close()
        return False


def check_blind(root: Path) -> list[Finding]:
    findings = _static_reads(root)
    findings.extend(_hash_intersection(root))
    findings.extend(check_committed_seals(root))
    if not opener_rejects_eval(root):
        findings.append(
            Finding(
                "blind-seal",
                "data/eval",
                "training opener allowed a read of data/eval; the read must be rejected",
            )
        )
    return findings


def check_committed_seals(root: Path) -> list[Finding]:
    """A payload under data/eval must match the seal committed beside it."""
    base = root / "data" / "eval"
    if not base.is_dir():
        return []
    findings: list[Finding] = []
    named: set[str] = set()
    for path in _files(base):
        if path.name.endswith(".sha256"):
            findings.extend(_check_sha256_sidecar(path, root, named))
        elif path.name == "seal.json":
            findings.extend(_check_seal_json(path, root, named))
    for path in _files(base):
        if not _payload_needs_seal(path):
            continue
        where = relpath(path, root)
        if where not in named:
            findings.append(Finding("blind-seal", where, "missing seal"))
    return findings


def _files(base: Path) -> list[Path]:
    return sorted(item for item in base.rglob("*") if item.is_file())


def _payload_needs_seal(path: Path) -> bool:
    if path.name == "seal.json":
        return False
    return path.suffix.lower() in {".json", ".jsonl"}


def _check_sha256_sidecar(path: Path, root: Path, named: set[str]) -> list[Finding]:
    where = relpath(path, root)
    try:
        parsed = _parse_sha256_lines(path.read_text(encoding="utf-8"))
    except UnicodeError:
        return [Finding("blind-seal", where, "sha256 seal is not UTF-8")]
    if parsed is None:
        return [Finding("blind-seal", where, "sha256 seal line is not 'digest filename'")]
    if not parsed:
        return [Finding("blind-seal", where, "sha256 seal lists no artifact")]
    findings: list[Finding] = []
    for digest, name in parsed:
        artifact = _resolve_artifact(path, name, root)
        if artifact is None:
            findings.append(
                Finding("blind-seal", where, f"seal artifact is outside data/eval: {name}")
            )
            continue
        _bind_artifact(artifact, digest, root, named, findings)
    return findings


def _check_seal_json(path: Path, root: Path, named: set[str]) -> list[Finding]:
    where = relpath(path, root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeError:
        return [Finding("blind-seal", where, "seal.json is not UTF-8")]
    except json.JSONDecodeError:
        return [Finding("blind-seal", where, "seal.json is not JSON")]
    if not isinstance(data, dict):
        return [Finding("blind-seal", where, "seal.json must be an object")]
    findings: list[Finding] = []
    algorithm = data.get("algorithm")
    artifact_name = data.get("artifact")
    digest = data.get("sha256")
    if algorithm != "sha256" or not isinstance(artifact_name, str) or not isinstance(digest, str):
        return [Finding("blind-seal", where, "seal.json must name a sha256 artifact")]
    if _DIGEST.fullmatch(digest) is None:
        return [Finding("blind-seal", where, "seal.json sha256 is not a digest")]
    artifact = _resolve_artifact(path, artifact_name, root)
    if artifact is None:
        return [
            Finding(
                "blind-seal",
                where,
                f"seal artifact is outside data/eval: {artifact_name}",
            )
        ]
    if not _bind_artifact(artifact, digest, root, named, findings):
        return findings
    if "count" in data:
        findings.extend(_check_count(artifact, data.get("count"), root))
    if "source_zip_sha256" in data:
        findings.extend(_check_source_zip(path, data.get("source_zip_sha256"), root))
    return findings


def _bind_artifact(
    artifact: Path,
    digest: str,
    root: Path,
    named: set[str],
    findings: list[Finding],
) -> bool:
    where = relpath(artifact, root)
    named.add(where)
    if not artifact.is_file():
        findings.append(Finding("blind-seal", where, "sealed artifact is missing"))
        return False
    actual = sha256_file(artifact)
    if actual.lower() != digest.lower():
        findings.append(
            Finding(
                "blind-seal",
                where,
                f"sha256 mismatch: seal {digest.lower()} file {actual}",
            )
        )
        return False
    return True


def _check_count(artifact: Path, count: object, root: Path) -> list[Finding]:
    where = relpath(artifact, root)
    if isinstance(count, bool) or not isinstance(count, int):
        return [Finding("blind-seal", where, "seal count is not an integer")]
    try:
        text = artifact.read_text(encoding="utf-8")
    except UnicodeError:
        return [Finding("blind-seal", where, "sealed artifact is not UTF-8")]
    actual = sum(1 for line in text.splitlines() if line.strip())
    if actual != count:
        return [
            Finding(
                "blind-seal",
                where,
                f"seal count mismatch: seal {count} file {actual}",
            )
        ]
    return []


def _check_source_zip(seal_path: Path, digest: object, root: Path) -> list[Finding]:
    where = relpath(seal_path, root)
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        return [Finding("blind-seal", where, "source_zip_sha256 is not a digest")]
    zip_path = locate(root, RAW_ZIP.as_posix(), "SLM_DATASETS_ZIP")
    if not zip_path.is_file():
        return [Finding("blind-seal", where, missing_message(RAW_ZIP.as_posix(), "SLM_DATASETS_ZIP"))]
    actual = sha256_file(zip_path)
    if actual.lower() != digest.lower():
        return [
            Finding(
                "blind-seal",
                where,
                f"source_zip_sha256 mismatch: seal {digest.lower()} archive {actual}",
            )
        ]
    return []


def _parse_sha256_lines(text: str) -> list[tuple[str, str]] | None:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2 or _DIGEST.fullmatch(parts[0]) is None:
            return None
        name = parts[1].strip()
        if name.startswith("*"):
            name = name[1:]
        if not name:
            return None
        rows.append((parts[0], name))
    return rows


def _resolve_artifact(seal_path: Path, name: str, root: Path) -> Path | None:
    relative = Path(name)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        return None
    candidate = (seal_path.parent / relative).resolve()
    eval_root = (root.resolve() / "data" / "eval").resolve()
    try:
        candidate.relative_to(eval_root)
    except ValueError:
        return None
    return candidate


def blind_summary(root: Path) -> str:
    sealed = seal_manifest(root)
    lines = [f"blind files hashed: {len(sealed)}"]
    lines.extend(f"  {item.digest}  {item.relpath}" for item in sealed)
    if _hash_intersection(root) or _static_reads(root):
        lines.append("training manifest ∩ blind ≠ ∅")
    else:
        lines.append("training manifest ∩ blind = ∅")
    if opener_rejects_eval(root):
        lines.append("training read attempt: REJECTED")
    else:
        lines.append("training read attempt: ALLOWED")
    return "\n".join(lines)


_EVAL_PATH = re.compile(r"data[/\\]+eval\b", re.IGNORECASE)
_MODULE_OPENERS = {"io", "builtins", "codecs"}
_PATH_MODULES = {"path", "posixpath", "ntpath"}


def _contains_eval_path(text: str) -> bool:
    return _EVAL_PATH.search(text) is not None


def _string_seq(node: ast.AST, env: dict[str, str]) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for elt in node.elts:
        folded = _fold(elt, env)
        if folded is None:
            return None
        values.append(folded)
    return values


def _fold_mod(node: ast.BinOp, env: dict[str, str]) -> str | None:
    template = _fold(node.left, env)
    if template is None:
        return None
    if isinstance(node.right, (ast.List, ast.Tuple)):
        values = _string_seq(node.right, env)
        if values is None:
            return None
        try:
            return template % tuple(values)
        except (TypeError, ValueError):
            return None
    value = _fold(node.right, env)
    if value is None:
        return None
    try:
        return template % value
    except (TypeError, ValueError):
        return None


def _is_path_join(func: ast.AST) -> bool:
    if not isinstance(func, ast.Attribute) or func.attr != "join":
        return False
    value = func.value
    if isinstance(value, ast.Name) and value.id in _PATH_MODULES:
        return True
    return isinstance(value, ast.Attribute) and value.attr == "path"


def _fold_call(node: ast.Call, env: dict[str, str]) -> str | None:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "Path" and len(node.args) == 1 and not node.keywords:
        return _fold(node.args[0], env)
    if isinstance(func, ast.Attribute) and func.attr == "format" and not node.keywords:
        template = _fold(func.value, env)
        if template is None:
            return None
        args: list[str] = []
        for arg in node.args:
            folded = _fold(arg, env)
            if folded is None:
                return None
            args.append(folded)
        try:
            return template.format(*args)
        except (IndexError, KeyError, ValueError):
            return None
    if isinstance(func, ast.Attribute) and func.attr == "join" and not node.keywords:
        if _is_path_join(func):
            parts: list[str] = []
            for arg in node.args:
                folded = _fold(arg, env)
                if folded is None:
                    return None
                parts.append(folded)
            if not parts:
                return None
            return "/".join(parts)
        receiver = _fold(func.value, env)
        if receiver is None or len(node.args) != 1:
            return None
        parts = _string_seq(node.args[0], env)
        if parts is None:
            return None
        return receiver.join(parts)
    return None


def _fold(node: ast.AST | None, env: dict[str, str]) -> str | None:
    """Constant string, including pieces joined into one path."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            if isinstance(value, ast.FormattedValue):
                folded = _fold(value.value, env)
                if folded is None:
                    return None
                parts.append(folded)
                continue
            return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left, env), _fold(node.right, env)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = _fold(node.left, env), _fold(node.right, env)
        if left is None or right is None:
            return None
        return f"{left.rstrip('/\\')}/{right.lstrip('/\\')}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _fold_mod(node, env)
    if isinstance(node, ast.Call):
        return _fold_call(node, env)
    return None


def _is_training_open(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "training_open"
    return isinstance(func, ast.Attribute) and func.attr == "training_open"


def _is_reader_expr(node: ast.AST | None, readers: set[str]) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name) and node.id in readers:
        return True
    if isinstance(node, ast.Attribute) and node.attr == "open":
        return isinstance(node.value, ast.Name) and node.value.id in _MODULE_OPENERS
    return False


def _is_raw_read(node: ast.Call, readers: set[str]) -> bool:
    if _is_training_open(node):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in readers
    if isinstance(func, ast.Attribute):
        if func.attr in {"read_text", "read_bytes"}:
            return True
        if func.attr == "open":
            return True
    return False


def _path_exprs(node: ast.Call) -> list[ast.AST]:
    func = node.func
    if isinstance(func, ast.Name):
        return [node.args[0]] if node.args else []
    if not isinstance(func, ast.Attribute):
        return []
    if func.attr in {"read_text", "read_bytes"}:
        return [func.value]
    if func.attr != "open":
        return []
    if isinstance(func.value, ast.Name) and func.value.id in _MODULE_OPENERS:
        return [node.args[0]] if node.args else []
    exprs = [func.value]
    if node.args:
        exprs.append(node.args[0])
    return exprs


def _proven_outside_eval(exprs: list[ast.AST], env: dict[str, str]) -> bool:
    if not exprs:
        return False
    for expr in exprs:
        folded = _fold(expr, env)
        if folded is None or _contains_eval_path(folded):
            return False
    return True


def _assigned_names(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Assign):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


class _TrainingReadScan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.env: dict[str, str] = {}
        self.readers = {"open"}
        self.eval_path = False
        self.raw_read = False

    def message(self) -> str | None:
        if self.eval_path:
            return "training code reads the sealed eval tree data/eval"
        if self.raw_read:
            return (
                "training code reads a file without training_open(); "
                "the read is not rejected"
            )
        return None

    def _note(self, text: str) -> None:
        if _contains_eval_path(text):
            self.eval_path = True

    def _bind(self, name: str, folded: str | None, reader: bool) -> None:
        if isinstance(folded, str):
            self.env[name] = folded
            self._note(folded)
        elif name in self.env:
            del self.env[name]
        if reader:
            self.readers.add(name)
        elif name in self.readers:
            self.readers.discard(name)

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.expr):
            folded = _fold(node, self.env)
            if isinstance(folded, str):
                self._note(folded)
        if isinstance(node, ast.Call) and _is_raw_read(node, self.readers):
            exprs = _path_exprs(node)
            if any(
                isinstance(folded, str) and _contains_eval_path(folded)
                for folded in (_fold(expr, self.env) for expr in exprs)
            ):
                self.eval_path = True
            elif not _proven_outside_eval(exprs, self.env):
                self.raw_read = True
        super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        saved_env = dict(self.env)
        saved_readers = set(self.readers)
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self.env.pop(arg.arg, None)
            self.readers.discard(arg.arg)
        if node.args.vararg is not None:
            self.env.pop(node.args.vararg.arg, None)
        if node.args.kwarg is not None:
            self.env.pop(node.args.kwarg.arg, None)
        self.generic_visit(node)
        self.env = saved_env
        self.readers = saved_readers

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        saved_env = dict(self.env)
        saved_readers = set(self.readers)
        self.generic_visit(node)
        self.env = saved_env
        self.readers = saved_readers

    def visit_Assign(self, node: ast.Assign) -> None:
        folded = _fold(node.value, self.env)
        reader = _is_reader_expr(node.value, self.readers)
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind(target.id, folded, reader)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        folded = _fold(node.value, self.env) if node.value is not None else None
        reader = _is_reader_expr(node.value, self.readers) if node.value is not None else False
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id, folded, reader)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        if not isinstance(node.target, ast.Name) or not isinstance(node.op, ast.Add):
            if isinstance(node.target, ast.Name):
                self.env.pop(node.target.id, None)
            return
        current = self.env.get(node.target.id)
        added = _fold(node.value, self.env)
        if current is not None and added is not None:
            self._bind(node.target.id, current + added, False)
        else:
            self.env.pop(node.target.id, None)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {None, "builtins", "io", "codecs"}:
            for alias in node.names:
                if alias.name == "open":
                    self.readers.add(alias.asname or alias.name)
        self.generic_visit(node)

    def _visit_branches(self, groups: list[list[ast.stmt]]) -> None:
        for statements in groups:
            for statement in statements:
                self.visit(statement)
        # A name written in a branch is not a constant afterwards.
        # Reader aliases stay, so a later call is still rejected.
        for name in set().union(*(_assigned_names(statements) for statements in groups)):
            self.env.pop(name, None)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_branches([node.body, node.orelse])

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._visit_branches([node.body, node.orelse])
        if isinstance(node.target, ast.Name):
            self.env.pop(node.target.id, None)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._visit_branches([node.body, node.orelse])

    def visit_Try(self, node: ast.Try) -> None:
        groups = [node.body, node.orelse, node.finalbody]
        groups.extend(handler.body for handler in node.handlers)
        self._visit_branches(groups)


def _python_read_message(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    scan = _TrainingReadScan()
    scan.visit(tree)
    return scan.message()


def _static_reads(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in training_code_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        regex_hit = any(pattern.search(text) for pattern in _READS)
        ast_message = _python_read_message(text) if path.suffix.lower() == ".py" else None
        if not regex_hit and ast_message is None:
            continue
        if regex_hit and (ast_message is None or "data/eval" not in ast_message):
            message = "training code reads the sealed eval tree data/eval"
        else:
            message = ast_message or "training code reads the sealed eval tree data/eval"
        findings.append(Finding("blind-seal", relpath(path, root), message))
    return findings


def _hash_intersection(root: Path) -> list[Finding]:
    blind: dict[str, list[str]] = {}
    for item in seal_manifest(root):
        blind.setdefault(item.digest, []).append(item.relpath)
    findings: list[Finding] = []
    for item in training_manifest(root):
        blind_paths = blind.get(item.digest)
        if not blind_paths:
            continue
        shared = ", ".join(blind_paths)
        findings.append(
            Finding(
                "blind-seal",
                item.relpath,
                f"training manifest ∩ blind manifest share sha256 {item.digest} with {shared}",
            )
        )
    return findings
