"""Shared grader result. Score is 1 when the case passes and 0 otherwise."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GradeResult:
    name: str
    passed: bool
    detail: str

    @property
    def score(self) -> float:
        return 1.0 if self.passed else 0.0


def pass_rate(results: list[GradeResult] | tuple[GradeResult, ...]) -> float:
    if not results:
        raise ValueError("pass_rate requires at least one result")
    return sum(result.score for result in results) / len(results)
