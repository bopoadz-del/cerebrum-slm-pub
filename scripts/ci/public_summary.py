"""Write public-CI counts and fail when a skip is not a private_data marker skip."""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _not_run_count(text: str, label: str) -> int:
    if label == "gates":
        lines = [line for line in text.splitlines() if line.startswith("NOT-RUN [")]
        green = "gates: green"
    else:
        lines = [line for line in text.splitlines() if line.startswith("NOT-RUN [collection-provenance]")]
        green = "provenance: green"
    for line in text.splitlines():
        if line.startswith(green + " (") and "not run: private data absent" in line:
            counted = int(line.split("(", 1)[1].split(" ", 1)[0])
            if counted != len(lines):
                raise SystemExit(f"{label} NOT-RUN count {counted} != {len(lines)} lines")
            return counted
    if any(line.strip() == green for line in text.splitlines()):
        if lines:
            raise SystemExit(f"{green} with NOT-RUN lines and no count")
        return 0
    return len(lines)


def _skip_message(case: ET.Element) -> str:
    skipped = case.find("skipped")
    if skipped is None:
        return ""
    message = skipped.get("message") or ""
    text = (skipped.text or "").strip()
    return message or text


def _is_private_data_skip(message: str) -> bool:
    text = message.strip()
    if text.startswith("Skipped:"):
        text = text[len("Skipped:") :].strip()
    return text.startswith("private_data:")


def summarize(gates_text: str, provenance_text: str, junit_path: Path | None) -> tuple[str, int]:
    passed = failed = skipped = errors = private_skips = other_skips = 0
    if junit_path is not None and junit_path.is_file():
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for suite in suites:
            passed += int(suite.get("tests", "0")) - int(suite.get("failures", "0")) - int(suite.get("errors", "0")) - int(suite.get("skipped", "0"))
            failed += int(suite.get("failures", "0"))
            errors += int(suite.get("errors", "0"))
            skipped += int(suite.get("skipped", "0"))
            for case in suite.findall("testcase"):
                message = _skip_message(case)
                if not message and case.find("skipped") is None:
                    continue
                if case.find("skipped") is None:
                    continue
                if _is_private_data_skip(message):
                    private_skips += 1
                else:
                    other_skips += 1
    gate_not_run = _not_run_count(gates_text, "gates") if gates_text else 0
    provenance_not_run = _not_run_count(provenance_text, "provenance") if provenance_text else 0
    lines = [
        "## Public CI counts",
        "",
        f"- pytest passed: {passed}",
        f"- pytest failed: {failed}",
        f"- pytest errors: {errors}",
        f"- pytest skipped: {skipped}",
        f"- private_data skips: {private_skips}",
        f"- other skips: {other_skips}",
        f"- gate NOT-RUN: {gate_not_run}",
        f"- provenance NOT-RUN: {provenance_not_run}",
        "",
    ]
    status = 0
    if other_skips:
        lines.append("A skip is not a private_data marker skip.")
        status = 1
    if junit_path is not None and junit_path.is_file() and skipped != private_skips + other_skips:
        lines.append("Skipped count does not match per-test skip records.")
        status = 1
    return "\n".join(lines) + "\n", status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Public CI count summary")
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args(argv)
    gates_text = args.gates.read_text(encoding="utf-8") if args.gates.is_file() else ""
    provenance_text = args.provenance.read_text(encoding="utf-8") if args.provenance.is_file() else ""
    junit = args.junit if args.junit.is_file() else None
    body, status = summarize(gates_text, provenance_text, junit)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(body)
    sys.stdout.write(body)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
