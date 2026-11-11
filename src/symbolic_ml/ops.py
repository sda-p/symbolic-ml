"""Counted primitive operations: the machine-independent measure of work."""


class Ops:
    """A mutable counter. Models and evaluators add to `n` as they work."""

    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0
