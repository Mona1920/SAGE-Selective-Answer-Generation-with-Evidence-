"""
Step 5: three-action decision policies.

Thresholds and classifiers are fit on train/validation only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .evidence import EvidenceFeatures, FEATURE_NAMES, to_vector

ACTIONS = ["ANSWER", "CLARIFY", "ABSTAIN"]


@dataclass
class Decision:
    action: str
    confidence: float
    reason: str
    probabilities: dict[str, float]


@dataclass
class RuleThresholds:
    ambiguity: float
    relevance: float
    margin: float
    support_min: float
    conflict: float


def rule_decide(feat: EvidenceFeatures, t: RuleThresholds) -> Decision:
    if feat.ambiguity >= t.ambiguity:
        return Decision(
            "CLARIFY",
            feat.ambiguity,
            "Question looks underspecified (pronouns/short/vague cues).",
            {"ANSWER": 0.1, "CLARIFY": float(feat.ambiguity), "ABSTAIN": 0.2},
        )
    if feat.conflict >= t.conflict:
        return Decision(
            "ABSTAIN",
            feat.conflict,
            "Retrieved passages support more than one incompatible answer.",
            {"ANSWER": 0.15, "CLARIFY": 0.15, "ABSTAIN": float(max(0.5, feat.conflict))},
        )
    if feat.retrieval_empty >= 0.5 or feat.max_relevance < t.relevance:
        return Decision(
            "ABSTAIN",
            1.0 - feat.max_relevance,
            "Retrieved evidence is weak or missing.",
            {"ANSWER": float(feat.max_relevance), "CLARIFY": 0.1, "ABSTAIN": 0.7},
        )
    if feat.goldish_support >= t.support_min and feat.option_margin >= t.margin:
        conf = 0.5 * feat.goldish_support + 0.3 * min(1.0, feat.max_relevance * 3) + 0.2 * min(1.0, feat.option_margin * 5)
        return Decision(
            "ANSWER",
            float(conf),
            "Evidence mentions a consistent answer with adequate retrieval score.",
            {"ANSWER": float(conf), "CLARIFY": 0.05, "ABSTAIN": float(1 - conf)},
        )
    return Decision(
        "ABSTAIN",
        0.6,
        "Evidence is related but does not sufficiently support one answer.",
        {"ANSWER": 0.25, "CLARIFY": 0.1, "ABSTAIN": 0.65},
    )


def tune_rule_thresholds(features: list[EvidenceFeatures], labels: list[str]) -> RuleThresholds:
    """Grid-search simple rules on validation labels only."""
    best = RuleThresholds(0.55, 0.08, 0.02, 0.5, 0.35)
    best_acc = -1.0
    for amb in (0.4, 0.5, 0.6, 0.7):
        for rel in (0.04, 0.08, 0.12, 0.18):
            for margin in (0.0, 0.02, 0.05):
                for conf in (0.3, 0.5, 0.8):
                    t = RuleThresholds(amb, rel, margin, 0.5, conf)
                    pred = [rule_decide(f, t).action for f in features]
                    acc = np.mean([p == y for p, y in zip(pred, labels)])
                    if acc > best_acc:
                        best_acc = acc
                        best = t
    return best


class CalibratedPolicy:
    """Multinomial logistic regression with probability calibration."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.base = LogisticRegression(max_iter=500, class_weight="balanced")
        self.model = None
        self.classes_ = ACTIONS

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        # cv=3 is small but keeps the validation/train sets usable.
        self.model = CalibratedClassifierCV(self.base, method="sigmoid", cv=3)
        self.model.fit(Xs, y)
        self.classes_ = list(self.model.classes_)

    def decide(self, feat: EvidenceFeatures) -> Decision:
        x = self.scaler.transform([to_vector(feat)])
        proba = self.model.predict_proba(x)[0]
        mapping = {c: float(p) for c, p in zip(self.classes_, proba)}
        for a in ACTIONS:
            mapping.setdefault(a, 0.0)
        action = max(mapping, key=mapping.get)
        conf = mapping[action]
        reasons = {
            "ANSWER": "Calibrated model assigns highest probability to ANSWER.",
            "CLARIFY": "Calibrated model assigns highest probability to CLARIFY.",
            "ABSTAIN": "Calibrated model assigns highest probability to ABSTAIN.",
        }
        extra = []
        if feat.ambiguity >= 0.5:
            extra.append(f"ambiguity={feat.ambiguity:.2f}")
        if feat.conflict >= 0.4:
            extra.append(f"conflict={feat.conflict:.2f}")
        if feat.goldish_support < 0.5:
            extra.append("answer not clearly mentioned in evidence")
        reason = reasons[action]
        if extra:
            reason += " Signals: " + ", ".join(extra) + "."
        return Decision(action, float(conf), reason, mapping)


class BinaryCalibratedAbstention:
    """Answer vs not-answer using calibrated P(correct-enough)."""

    def __init__(self, answer_threshold: float = 0.5):
        self.threshold = answer_threshold
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X: np.ndarray, answerable: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        base = LogisticRegression(max_iter=500, class_weight="balanced")
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self.model.fit(Xs, answerable)

    def decide(self, feat: EvidenceFeatures) -> Decision:
        x = self.scaler.transform([to_vector(feat)])
        p = float(self.model.predict_proba(x)[0, 1])
        if p >= self.threshold:
            return Decision(
                "ANSWER",
                p,
                f"Calibrated P(answerable)={p:.2f} >= {self.threshold:.2f}.",
                {"ANSWER": p, "CLARIFY": 0.0, "ABSTAIN": 1 - p},
            )
        return Decision(
            "ABSTAIN",
            1 - p,
            f"Calibrated P(answerable)={p:.2f} < {self.threshold:.2f}.",
            {"ANSWER": p, "CLARIFY": 0.0, "ABSTAIN": 1 - p},
        )


def tune_binary_threshold(policy: BinaryCalibratedAbstention, X, y_answerable, features, objective="f1"):
    """Pick a threshold on validation. Never call this on test labels."""
    best_t, best_s = 0.5, -1.0
    ps = policy.model.predict_proba(policy.scaler.transform(X))[:, 1]
    for t in np.linspace(0.2, 0.8, 13):
        pred = (ps >= t).astype(int)
        tp = int(((pred == 1) & (y_answerable == 1)).sum())
        fp = int(((pred == 1) & (y_answerable == 0)).sum())
        fn = int(((pred == 0) & (y_answerable == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_s:
            best_s, best_t = f1, float(t)
    policy.threshold = best_t
    return best_t, best_s


def feature_matrix(feats: list[EvidenceFeatures]) -> np.ndarray:
    return np.array([to_vector(f) for f in feats], dtype=float)
