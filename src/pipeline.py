"""End-to-end processing for one question."""

from __future__ import annotations

from .dataset import Example
from .evidence import extract_features
from .generate import generate_answer
from .retrieve import TfidfRetriever
from .utils import Timer


def process_example(example: Example, retriever: TfidfRetriever, top_k: int) -> dict:
    timer = Timer()
    extras = None
    if example.injected_conflict_passage:
        extras = [(f"conflict-{example.example_id}", example.injected_conflict_passage)]
    hits = retriever.search(
        example.question,
        top_k=top_k,
        exclude_ids=example.withhold_passage_ids,
        extra_passages=extras,
    )
    gen = generate_answer(example.question, example.options, hits)
    feat = extract_features(example.question, hits, gen)
    return {
        "example_id": example.example_id,
        "split": example.split,
        "question": example.question,
        "options": example.options,
        "gold_answer": example.gold_answer,
        "gold_action": example.gold_action,
        "evidence_type": example.evidence_type,
        "gold_passage_id": example.gold_passage_id,
        "construction_note": example.construction_note,
        "clarification_target": example.clarification_target,
        "predicted_answer": gen.predicted_answer,
        "option_scores": gen.option_scores,
        "extractive_span": gen.extractive_span,
        "supported_by_context": gen.supported_by_context,
        "retrieved": [
            {"passage_id": h.passage_id, "source": h.source, "score": h.score, "text": h.text}
            for h in hits
        ],
        "retrieved_ids": [h.passage_id for h in hits],
        "features": feat.__dict__,
        "latency_ms": timer.ms(),
        "_feat": feat,
        "_gen": gen,
    }
