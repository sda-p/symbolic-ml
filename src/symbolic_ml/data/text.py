"""Natural-text corpora, character level."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .synthetic import Sample

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
URLS = {
    "tinyshakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def load_chars(name: str = "tinyshakespeare") -> Sample:
    """R1: the corpus as a sequence of character ids (ids sorted by character)."""
    path = RAW / name / "input.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URLS[name], path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    index = {c: i for i, c in enumerate(chars)}
    seq = np.array([index[c] for c in text], dtype=np.int64)
    return Sample(seq, np.full(len(seq), np.nan), len(chars), chars)
