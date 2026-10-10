"""
Step 4: measurable evidence features.

Semantic similarity is one feature, not proof of support.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .generate import Generation, conflict_score, option_margin
from .retrieve import Retrieved
from .utils import answer_mentioned, normalize_text

AMBIGUITY_CUES = (
    "this",
    "that",
    "it",
    "they",
    "these",
    "those",
    "the process",
    "the phenomenon",
    "which one",
    "what happens",
    "what is this",
    "the answer",
)


@dataclass
class EvidenceFeatures:
    max_relevance: float
    mean_relevance: float
    goldish_support: float  # predicted answer mentioned in context
    option_margin: float
    conflict: float
    ambiguity: float
    retrieval_empty: float
    n_hits: float


def ambiguity_score(question: str) -> float:
    q = normalize_text(question)
    tokens = q.split()
    score = 0.0
    if len(tokens) <= 6:
        score += 0.35
    for cue in AMBIGUITY_CUES:
        if cue in q:
            score += 0.25
    if re.search(r"\b(this|it|that)\b", q):
        score += 0.2
    if q.endswith("?") and len(tokens) <= 8:
        score += 0.1
    return min(1.0, score)


def extract_features(question: str, hits: list[Retrieved], gen: Generation) -> EvidenceFeatures:
    rel = [h.score for h in hits] or [0.0]
    context = " ".join(h.text for h in hits)
    support = 1.0 if answer_mentioned(gen.predicted_answer, context) else 0.0
    # Partial credit if any option is mentioned, but predicted is not.
    if support == 0.0 and any(answer_mentioned(opt, context) for opt in gen.option_scores):
        support = 0.25
    return EvidenceFeatures(
        max_relevance=float(max(rel)),
        mean_relevance=float(sum(rel) / len(rel)),
        goldish_support=support,
        option_margin=option_margin(gen.option_scores),
        conflict=conflict_score(gen.option_scores, hits, list(gen.option_scores.keys())),
        ambiguity=ambiguity_score(question),
        retrieval_empty=1.0 if not hits or max(rel) <= 1e-9 else 0.0,
        n_hits=float(len(hits)),
    )


FEATURE_NAMES = [
    "max_relevance",
    "mean_relevance",
    "goldish_support",
    "option_margin",
    "conflict",
    "ambiguity",
    "retrieval_empty",
    "n_hits",
]


def to_vector(feat: EvidenceFeatures) -> list[float]:
    d = asdict(feat)
    return [d[name] for name in FEATURE_NAMES]
