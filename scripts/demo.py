#!/usr/bin/env python3
"""Interactive CLI demo of ANSWER / CLARIFY / ABSTAIN."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import Example, load_sciq, construct_all
from src.evidence import extract_features
from src.generate import generate_answer
from src.pipeline import process_example
from src.policy import CalibratedPolicy, feature_matrix
from src.retrieve import TfidfRetriever
from src.utils import set_seed
import numpy as np
import yaml


def train_once():
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    sciq = load_sciq()
    passages, splits = construct_all(sciq, cfg["n_per_class"], cfg["seed"])
    retriever = TfidfRetriever(cfg["max_features_tfidf"], tuple(cfg["ngram_range"]))
    retriever.fit(passages)
    train_recs = [process_example(ex, retriever, cfg["top_k"]) for ex in splits["train"]]
    X = feature_matrix([r["_feat"] for r in train_recs])
    y = np.array([r["gold_action"] for r in train_recs])
    policy = CalibratedPolicy()
    policy.fit(X, y)
    return retriever, policy, cfg


def show(question: str, retriever, policy, cfg, options=None):
    options = options or ["yes", "no", "unknown", "not enough information"]
    ex = Example(
        example_id="demo",
        split="demo",
        question=question,
        options=options,
        gold_answer="unknown",
        gold_action="ANSWER",
        evidence_type="sufficient",
        gold_passage_id=None,
        withhold_passage_ids=[],
        injected_conflict_passage=None,
        construction_note="interactive",
        clarification_target=None,
    )
    rec = process_example(ex, retriever, cfg["top_k"])
    dec = policy.decide(rec["_feat"])
    print("\nQuestion:", question)
    print("Action:", dec.action, f"(confidence {dec.confidence:.3f} — this is calibrated model probability, not certainty)")
    print("Reason:", dec.reason)
    print("Evidence features:", json.dumps(rec["features"], indent=2))
    print("Retrieved:")
    for h in rec["retrieved"][:3]:
        print(f"  [{h['passage_id']}] score={h['score']:.3f} {h['text'][:220]}...")
    if dec.action == "ANSWER":
        print("Answer:", rec["predicted_answer"])
        print("Extracted span:", rec["extractive_span"][:300])
    elif dec.action == "CLARIFY":
        print("Clarification: Which specific scientific entity or process do you mean?")
    else:
        print("Abstention:", dec.reason)
    print("Latency ms:", round(rec["latency_ms"], 1))


if __name__ == "__main__":
    print("Training a lightweight policy once (SciQ TF-IDF RAG)...")
    retriever, policy, cfg = train_once()
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]), retriever, policy, cfg)
    else:
        print("Enter questions. Empty line to quit.")
        while True:
            q = input("\nQ> ").strip()
            if not q:
                break
            show(q, retriever, policy, cfg)
