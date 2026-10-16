"""Shared stop for baselines that are not sampled again."""


class BaselineNotRun(RuntimeError):
    """The untuned and prompt-only baselines are not sampled from this package."""
