"""Small helpers used across the project."""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Make sampling and sklearn as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(text: str) -> str:
    """Lowercase and collapse punctuation so string matching is less brittle."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_mentioned(answer: str, context: str) -> bool:
    """True if the answer string appears in the retrieved context."""
    a = normalize_text(answer)
    c = normalize_text(context)
    if not a or not c:
        return False
    return a in c


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Timer:
    """Wall-clock timer for latency measurements."""

    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0
