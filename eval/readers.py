"""Read eval inputs. Sealed files go through the record reader, never training_open."""

from __future__ import annotations

from pathlib import Path

from gates.blind_seal import check_committed_seals, path_is_sealed
from gates.records import RecordParseError, load_records
from private_data import locate, missing_message


class EvalReadError(RuntimeError):
    """An eval file could not be read. The message has no item text."""


def read_records(root: Path, relative: str) -> list[tuple[int, dict]]:
    """Return (line, object) rows for one eval artifact.

    A path under data/eval is sealed. This function checks that seal, then
    reads with load_records. It does not call the training opener.
    """
    path = locate(root, relative)
    if not path.is_file():
        raise EvalReadError(missing_message(relative))
    try:
        path.resolve().relative_to(root.resolve())
        in_repo = True
    except ValueError:
        in_repo = False
    sealed_target = path if in_repo else (root / relative)
    sealed = path_is_sealed(sealed_target, root)
    if not sealed and _looks_sealed(relative):
        raise EvalReadError("sealed eval path was not accepted")
    if sealed and in_repo:
        _require_seal(root, relative)
    try:
        return load_records(path)
    except RecordParseError as exc:
        raise EvalReadError(f"eval record failed to parse at line {exc.line}") from None
    except UnicodeError as exc:
        raise EvalReadError(f"eval artifact is not UTF-8: {relative}") from exc
    except OSError as exc:
        raise EvalReadError(f"eval artifact is missing: {relative}") from exc


def _looks_sealed(relative: str) -> bool:
    text = relative.replace("\\", "/")
    return "data/eval" in text


def _require_seal(root: Path, relative: str) -> None:
    watched = {
        relative,
        relative + ".sha256",
        str(Path(relative).parent / "seal.json"),
    }
    findings = check_committed_seals(root)
    if any(finding.path in watched for finding in findings):
        raise EvalReadError("sealed eval artifact failed its seal check")
