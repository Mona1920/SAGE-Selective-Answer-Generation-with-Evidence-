"""
Step 3 retrieval: TF-IDF + cosine search.

Sentence-Transformers / FAISS were planned for Colab. On this machine the
available Python is 3.14, which currently lacks reliable PyTorch/FAISS
wheels, so lexical TF-IDF is the implemented retriever. It is a legitimate
sparse RAG baseline and stays reproducible without paid APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import Passage


@dataclass
class Retrieved:
    passage_id: str
    text: str
    source: str
    score: float


class TfidfRetriever:
    def __init__(self, max_features: int = 30000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=1,
        )
        self.passages: list[Passage] = []
        self.matrix = None
        self.id_to_index: dict[str, int] = {}

    def fit(self, passages: list[Passage]) -> None:
        self.passages = passages
        texts = [p.text for p in passages]
        self.matrix = self.vectorizer.fit_transform(texts)
        self.id_to_index = {p.passage_id: i for i, p in enumerate(passages)}

    def search(
        self,
        query: str,
        top_k: int = 5,
        exclude_ids: list[str] | None = None,
        extra_passages: list[tuple[str, str]] | None = None,
    ) -> list[Retrieved]:
        """
        extra_passages: list of (passage_id, text) injected only for this query,
        used to simulate contradictory evidence.
        """
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        exclude = set(exclude_ids or [])
        ranked = np.argsort(-scores)
        hits: list[Retrieved] = []
        for idx in ranked:
            pid = self.passages[idx].passage_id
            if pid in exclude:
                continue
            hits.append(
                Retrieved(
                    passage_id=pid,
                    text=self.passages[idx].text,
                    source=self.passages[idx].source,
                    score=float(scores[idx]),
                )
            )
            if len(hits) >= top_k:
                break

        if extra_passages:
            extra_texts = [t for _, t in extra_passages]
            extra_mat = self.vectorizer.transform(extra_texts)
            extra_scores = cosine_similarity(q, extra_mat).ravel()
            extras = [
                Retrieved(
                    passage_id=pid,
                    text=text,
                    source="injected_conflict",
                    score=float(sc),
                )
                for (pid, text), sc in zip(extra_passages, extra_scores)
            ]
            # Keep injected conflicts even if lexical similarity is low.
            merged = extras + [h for h in hits if h.passage_id not in {e.passage_id for e in extras}]
            merged.sort(key=lambda h: h.score, reverse=True)
            hits = merged[: max(top_k, len(extras))]
        return hits
