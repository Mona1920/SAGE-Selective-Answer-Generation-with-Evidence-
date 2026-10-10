"""Metrics. Every function returns numbers computed from predictions, never placeholders."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def accuracy(y_true, y_pred) -> float:
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m.tolist()


def precision_recall(y_true, y_pred, positive: str):
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def coverage(actions) -> float:
    return float(np.mean([a == "ANSWER" for a in actions]))


def unsupported_answer_rate(records) -> float:
    """
    Fraction of all questions where the system answered without being allowed
    to, or answered with the wrong option.
    """
    n = len(records)
    bad = 0
    for r in records:
        if r["pred_action"] != "ANSWER":
            continue
        if r["gold_action"] != "ANSWER":
            bad += 1
        elif r["predicted_answer"].strip().lower() != r["gold_answer"].strip().lower():
            bad += 1
    return bad / max(n, 1)


def false_abstention_rate(records) -> float:
    """Gold ANSWER items on which the system abstained."""
    gold_answer = [r for r in records if r["gold_action"] == "ANSWER"]
    if not gold_answer:
        return 0.0
    return np.mean([r["pred_action"] == "ABSTAIN" for r in gold_answer])


def selective_risk(records) -> float:
    answered = [r for r in records if r["pred_action"] == "ANSWER"]
    if not answered:
        return 0.0
    return np.mean(
        [
            r["predicted_answer"].strip().lower() != r["gold_answer"].strip().lower()
            or r["gold_action"] != "ANSWER"
            for r in answered
        ]
    )


def exact_match_among_answered(records) -> float:
    answered = [r for r in records if r["pred_action"] == "ANSWER"]
    if not answered:
        return 0.0
    return np.mean(
        [
            r["predicted_answer"].strip().lower() == r["gold_answer"].strip().lower()
            for r in answered
        ]
    )


def citation_hit(records) -> float:
    """For gold ANSWER items, was the gold passage retrieved?"""
    items = [r for r in records if r["gold_action"] == "ANSWER" and r.get("gold_passage_id")]
    if not items:
        return 0.0
    return np.mean([r["gold_passage_id"] in r.get("retrieved_ids", []) for r in items])


def brier_multiclass(y_true, probas, labels) -> float:
    idx = {l: i for i, l in enumerate(labels)}
    s = 0.0
    for y, p in zip(y_true, probas):
        onehot = np.zeros(len(labels))
        if y in idx:
            onehot[idx[y]] = 1.0
        vec = np.array([p.get(l, 0.0) for l in labels])
        s += float(np.sum((vec - onehot) ** 2))
    return s / max(len(y_true), 1)


def ece(confidences, correct, n_bins=10) -> float:
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if i == 0:
            m = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if not np.any(m):
            continue
        acc = correct[m].mean()
        conf = confidences[m].mean()
        ece_val += (m.mean()) * abs(acc - conf)
    return float(ece_val)


def reliability_points(confidences, correct, n_bins=10):
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if i == 0:
            m = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        if not np.any(m):
            continue
        xs.append(float(confidences[m].mean()))
        ys.append(float(correct[m].mean()))
        ns.append(int(m.sum()))
    return xs, ys, ns


def risk_coverage_curve(records, score_key="answer_score"):
    """
    Sort by score-of-answering descending and compute risk at each coverage
    if we answered the top-ranked items.
    """
    ranked = sorted(records, key=lambda r: r.get(score_key, 0.0), reverse=True)
    risks, covs = [], []
    errors = 0
    for i, r in enumerate(ranked, start=1):
        wrong = (
            r["predicted_answer"].strip().lower() != r["gold_answer"].strip().lower()
            or r["gold_action"] != "ANSWER"
        )
        errors += int(wrong)
        risks.append(errors / i)
        covs.append(i / len(ranked))
    return covs, risks


def summarize(records, labels=("ANSWER", "CLARIFY", "ABSTAIN")) -> dict:
    y_true = [r["gold_action"] for r in records]
    y_pred = [r["pred_action"] for r in records]
    probas = [r.get("probabilities", {}) for r in records]
    confs = [r.get("confidence", 0.0) for r in records]
    correct_action = [int(a == b) for a, b in zip(y_true, y_pred)]
    return {
        "n": len(records),
        "action_accuracy": accuracy(y_true, y_pred),
        "confusion": confusion(y_true, y_pred, labels),
        "labels": list(labels),
        "coverage": coverage(y_pred),
        "unsupported_answer_rate": unsupported_answer_rate(records),
        "false_abstention_rate": float(false_abstention_rate(records)),
        "selective_risk": float(selective_risk(records)),
        "em_among_answered": float(exact_match_among_answered(records)),
        "clarify": precision_recall(y_true, y_pred, "CLARIFY"),
        "abstain": precision_recall(y_true, y_pred, "ABSTAIN"),
        "answer": precision_recall(y_true, y_pred, "ANSWER"),
        "citation_hit_gold_answer_items": float(citation_hit(records)),
        "brier": brier_multiclass(y_true, probas, labels),
        "ece": ece(confs, correct_action),
        "mean_latency_ms": float(np.mean([r.get("latency_ms", 0.0) for r in records])),
        "gold_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),
        "by_evidence_type": _by_type(records),
    }


def _by_type(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["evidence_type"]].append(r)
    out = {}
    for k, rs in groups.items():
        y_true = [r["gold_action"] for r in rs]
        y_pred = [r["pred_action"] for r in rs]
        out[k] = {
            "n": len(rs),
            "action_accuracy": accuracy(y_true, y_pred),
            "unsupported_answer_rate": unsupported_answer_rate(rs),
            "coverage": coverage(y_pred),
        }
    return out
