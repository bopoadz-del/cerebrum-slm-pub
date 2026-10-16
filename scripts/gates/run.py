"""CI entrypoint for the lane-1 repo gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gates.augment import check_augmentation
from gates.blind_seal import blind_summary, check_blind
from gates.findings import Finding
from gates.held_out_shards import check_held_out_shards
from gates.no_hardwiring import check_hardwiring
from gates.prove import ProofError, run_proofs
from gates.raw_manifest import check_raw
from gates.split import check_split


def evaluate(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for checker in (
        check_raw,
        check_hardwiring,
        check_split,
        check_blind,
        check_augmentation,
        check_held_out_shards,
    ):
        findings.extend(checker(root))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lane 1 repo gates")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--skip-proof",
        action="store_true",
        help="Skip the red/remove/green planting proof",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    findings = evaluate(root)
    print(blind_summary(root))
    if not args.skip_proof:
        try:
            for line in run_proofs():
                print(line)
        except ProofError as exc:
            findings.append(Finding("proof", "", str(exc)))
    if findings:
        for finding in findings:
            print(finding.format(), file=sys.stderr)
        print(f"{len(findings)} gate failure(s)", file=sys.stderr)
        return 1
    print("gates: green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
