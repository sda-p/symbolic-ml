"""Prequential (predict-then-update) evaluation and result logging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"

# Every model is scored from this position on, so that models with different
# context depths are compared on exactly the same tokens.
START = 16


def prequential(model, seq: np.ndarray, start: int = START) -> np.ndarray:
    """Per-token loss in bits for seq[start:], updating after each prediction."""
    s = seq.tolist()
    step = model.step
    return np.array([step(s, t) for t in range(start, len(s))])


def summarize(bits: np.ndarray, tail: float = 0.1) -> dict:
    """Mean loss over the whole stream and over its final `tail` fraction."""
    k = max(1, int(len(bits) * tail))
    return {"bits": float(bits.mean()), "tail_bits": float(bits[-k:].mean())}


def write_jsonl(name: str, rows: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path
