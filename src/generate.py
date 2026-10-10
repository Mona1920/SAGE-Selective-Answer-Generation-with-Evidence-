"""Answer generation from retrieved context only (no parametric guessing)."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .retrieve import Retrieved
from .utils import answer_mentioned, normalize_text


@dataclass
class Generation:
    predicted_answer: str
    option_scores: dict[str, float]
    extractive_span: str
    supported_by_context: bool


def generate_answer(question: str, options: list[str], hits: list[Retrieved]) -> Generation:
    """
    Score each multiple-choice option against retrieved text with TF-IDF cosine.
    This is a lightweight extractive reader, not a large language model.
    """
    context = " ".join(h.text for h in hits) if hits else ""
    if not context.strip():
        scores = {opt: 0.0 for opt in options}
        return Generation(
            predicted_answer=options[0],
            option_scores=scores,
            extractive_span="",
            supported_by_context=False,
        )

    docs = [context] + options
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(docs)
    sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
    scores = {opt: float(s) for opt, s in zip(options, sims)}
    predicted = max(scores, key=scores.get)

    sentences = [s.strip() for s in context.replace("!", ".").split(".") if s.strip()]
    span = ""
    if sentences:
        sent_mat = vec.transform(sentences)
        pred_mat = vec.transform([predicted])
        best = int(cosine_similarity(pred_mat, sent_mat).ravel().argmax())
        span = sentences[best]

    return Generation(
        predicted_answer=predicted,
        option_scores=scores,
        extractive_span=span,
        supported_by_context=answer_mentioned(predicted, context),
    )


def option_margin(scores: dict[str, float]) -> float:
    vals = sorted(scores.values(), reverse=True)
    if len(vals) < 2:
        return float(vals[0]) if vals else 0.0
    return float(vals[0] - vals[1])


def conflict_score(scores: dict[str, float], hits: list[Retrieved], options: list[str]) -> float:
    """
    High when two different options are each mentioned in retrieved evidence
    or when the top two option similarities are close.
    """
    mentioned = [opt for opt in options if any(answer_mentioned(opt, h.text) for h in hits)]
    mention_conflict = 1.0 if len(set(normalize_text(m) for m in mentioned)) >= 2 else 0.0
    margin = option_margin(scores)
    closeness = max(0.0, 0.15 - margin) / 0.15
    return max(mention_conflict, closeness * 0.5)
