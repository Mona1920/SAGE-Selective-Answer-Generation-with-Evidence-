#!/usr/bin/env python3
"""Run Steps 2-7: construct data, retrieve, decide, compare, plot, save metrics."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import construct_all, examples_to_json, load_sciq
from src.evaluate import reliability_points, risk_coverage_curve, summarize
from src.evidence import EvidenceFeatures, FEATURE_NAMES
from src.pipeline import process_example
from src.plots import plot_ablation, plot_confusion, plot_metric_bars, plot_reliability, plot_risk_coverage
from src.policy import (
    ACTIONS,
    BinaryCalibratedAbstention,
    CalibratedPolicy,
    RuleThresholds,
    feature_matrix,
    rule_decide,
    tune_binary_threshold,
    tune_rule_thresholds,
)
from src.retrieve import TfidfRetriever
from src.utils import save_json, set_seed


def attach_decision(rec: dict, decision) -> dict:
    out = {k: v for k, v in rec.items() if not k.startswith("_")}
    out["pred_action"] = decision.action
    out["confidence"] = decision.confidence
    out["reason"] = decision.reason
    out["probabilities"] = decision.probabilities
    out["answer_score"] = decision.probabilities.get("ANSWER", 0.0)
    if decision.action == "CLARIFY":
        out["response"] = (
            "Your question is missing essential detail. "
            "Which specific process, entity, or condition do you mean?"
        )
        if rec.get("clarification_target"):
            out["response"] += " (A more specific version would name the original subject.)"
    elif decision.action == "ABSTAIN":
        out["response"] = f"I cannot give a reliable answer. {decision.reason}"
    else:
        cites = ", ".join(rec.get("retrieved_ids", [])[:3]) or "none"
        out["response"] = f"{rec['predicted_answer']} (citations: {cites})"
    return out


def always_answer(rec):
    from src.policy import Decision

    p = 0.99
    return Decision("ANSWER", p, "Standard RAG always answers.", {"ANSWER": p, "CLARIFY": 0.0, "ABSTAIN": 0.0})


def fixed_threshold_policy(feat, t: float):
    from src.policy import Decision

    if feat.max_relevance >= t:
        return Decision(
            "ANSWER",
            min(1.0, feat.max_relevance / max(t, 1e-6)),
            f"Fixed retrieval threshold: max_relevance {feat.max_relevance:.3f} >= {t:.3f}.",
            {"ANSWER": float(feat.max_relevance), "CLARIFY": 0.0, "ABSTAIN": float(1 - feat.max_relevance)},
        )
    return Decision(
        "ABSTAIN",
        1.0 - feat.max_relevance,
        f"Fixed retrieval threshold: max_relevance {feat.max_relevance:.3f} < {t:.3f}.",
        {"ANSWER": float(feat.max_relevance), "CLARIFY": 0.0, "ABSTAIN": float(1 - feat.max_relevance)},
    )


def tune_fixed_threshold(val_recs) -> float:
    y = np.array([r["gold_action"] == "ANSWER" for r in val_recs], dtype=int)
    rel = np.array([r["_feat"].max_relevance for r in val_recs])
    best_t, best_f1 = 0.1, -1.0
    for t in np.linspace(0.02, 0.4, 20):
        pred = (rel >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def apply_policy(records, fn):
    return [attach_decision(r, fn(r)) for r in records]


def zero_feature(feat: EvidenceFeatures, name: str) -> EvidenceFeatures:
    d = dict(feat.__dict__)
    d[name] = 0.0
    return EvidenceFeatures(**d)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    out = ROOT / cfg["output_dir"]
    data_dir = ROOT / cfg["data_dir"]
    fig_dir = out / "figures"
    out.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    print("Loading SciQ...")
    sciq = load_sciq()
    print({k: len(sciq[k]) for k in sciq})

    passages, splits = construct_all(sciq, cfg["n_per_class"], cfg["seed"])
    print("Corpus passages:", len(passages))
    for name, exs in splits.items():
        print(name, len(exs), {a: sum(e.gold_action == a for e in exs) for a in ACTIONS})

    save_json(data_dir / "label_definitions.json", {
        "ANSWER": "Question is specific and gold supporting evidence remains in the corpus.",
        "CLARIFY": "Question was underspecified; a targeted user question is required.",
        "ABSTAIN": "Evidence is missing (gold passage withheld/empty) or contradictory (distractor asserted).",
        "note": "Labels are constructed. Original SciQ has no three-action labels.",
        "n_per_class": cfg["n_per_class"],
        "seed": cfg["seed"],
    })
    for name, exs in splits.items():
        save_json(data_dir / f"{name}.json", examples_to_json(exs))

    print("Fitting TF-IDF retriever...")
    retriever = TfidfRetriever(cfg["max_features_tfidf"], tuple(cfg["ngram_range"]))
    retriever.fit(passages)

    def run_split(examples):
        recs = []
        for i, ex in enumerate(examples):
            recs.append(process_example(ex, retriever, cfg["top_k"]))
            if (i + 1) % 50 == 0:
                print(f"  processed {i+1}/{len(examples)}")
        return recs

    print("Processing train/val/test through retrieval + reader...")
    processed = {name: run_split(exs) for name, exs in splits.items()}

    train_feats = [r["_feat"] for r in processed["train"]]
    val_feats = [r["_feat"] for r in processed["validation"]]
    y_train = np.array([r["gold_action"] for r in processed["train"]])
    y_val = np.array([r["gold_action"] for r in processed["validation"]])
    X_train = feature_matrix(train_feats)
    X_val = feature_matrix(val_feats)

    print("Tuning rule thresholds on validation...")
    rules = tune_rule_thresholds(val_feats, list(y_val))
    print("Chosen rules:", rules)

    print("Tuning fixed retrieval threshold on validation...")
    fixed_t = tune_fixed_threshold(processed["validation"])
    print("Fixed threshold:", fixed_t)

    print("Fitting calibrated 3-class policy on train...")
    three = CalibratedPolicy()
    three.fit(X_train, y_train)

    print("Fitting binary calibrated abstention on train...")
    binary = BinaryCalibratedAbstention()
    y_bin_train = (y_train == "ANSWER").astype(int)
    y_bin_val = (y_val == "ANSWER").astype(int)
    binary.fit(X_train, y_bin_train)
    bt, bf1 = tune_binary_threshold(binary, X_val, y_bin_val, val_feats)
    print("Binary threshold:", bt, "val F1:", bf1)

    save_json(out / "thresholds.json", {
        "rule": rules.__dict__,
        "fixed_retrieval_threshold": fixed_t,
        "binary_answer_threshold": bt,
        "binary_val_f1": bf1,
        "tuned_on": "validation",
        "never_tuned_on": "test",
    })

    methods = {
        "standard_rag": lambda r: always_answer(r),
        "fixed_threshold": lambda r: fixed_threshold_policy(r["_feat"], fixed_t),
        "calibrated_binary": lambda r: binary.decide(r["_feat"]),
        "rule_three_action": lambda r: rule_decide(r["_feat"], rules),
        "proposed_calibrated_three_action": lambda r: three.decide(r["_feat"]),
    }

    test_summaries = {}
    test_outputs = {}
    for name, fn in methods.items():
        recs = apply_policy(processed["test"], fn)
        test_outputs[name] = recs
        test_summaries[name] = summarize(recs)
        save_json(out / f"predictions_{name}.json", recs)
        print(f"\n=== {name} ===")
        s = test_summaries[name]
        for k in [
            "action_accuracy",
            "unsupported_answer_rate",
            "false_abstention_rate",
            "coverage",
            "selective_risk",
            "em_among_answered",
            "citation_hit_gold_answer_items",
            "brier",
            "ece",
            "mean_latency_ms",
        ]:
            print(f"  {k}: {s[k]:.4f}")
        print("  clarify:", s["clarify"])

    save_json(out / "test_metrics.json", test_summaries)

    # Ablations of the proposed model
    print("\nAblations...")
    ablations = {"full": test_summaries["proposed_calibrated_three_action"]}
    for drop in ["ambiguity", "conflict", "goldish_support", "max_relevance"]:
        dropped = [i for i, n in enumerate(FEATURE_NAMES) if n == drop][0]
        Xtr = X_train.copy()
        Xtr[:, dropped] = 0.0
        model = CalibratedPolicy()
        model.fit(Xtr, y_train)

        def fn(r, m=model, d=drop):
            return m.decide(zero_feature(r["_feat"], d))

        recs = apply_policy(processed["test"], fn)
        ablations[f"no_{drop}"] = summarize(recs)
    save_json(out / "ablations.json", ablations)

    # Plots
    plot_metric_bars(test_summaries, fig_dir / "metrics_comparison.png")
    plot_confusion(
        test_summaries["proposed_calibrated_three_action"]["confusion"],
        test_summaries["proposed_calibrated_three_action"]["labels"],
        "Proposed 3-action policy",
        fig_dir / "confusion_proposed.png",
    )
    plot_confusion(
        test_summaries["standard_rag"]["confusion"],
        test_summaries["standard_rag"]["labels"],
        "Standard RAG (always answer)",
        fig_dir / "confusion_standard_rag.png",
    )
    prop = test_outputs["proposed_calibrated_three_action"]
    xs, ys, ns = reliability_points(
        [r["confidence"] for r in prop],
        [int(r["pred_action"] == r["gold_action"]) for r in prop],
    )
    plot_reliability(xs, ys, ns, test_summaries["proposed_calibrated_three_action"]["ece"], fig_dir / "reliability.png")

    curves = {}
    for name, recs in test_outputs.items():
        curves[name] = risk_coverage_curve(recs, "answer_score")
    plot_risk_coverage(curves, fig_dir / "risk_coverage.png")
    plot_ablation(ablations, fig_dir / "ablations.png")

    # Error analysis samples
    errors = {
        "answered_without_support": [
            {k: r[k] for k in ("example_id", "question", "gold_action", "pred_action", "predicted_answer", "reason", "evidence_type")}
            for r in prop
            if r["pred_action"] == "ANSWER" and r["gold_action"] != "ANSWER"
        ][:15],
        "false_abstention": [
            {k: r[k] for k in ("example_id", "question", "gold_action", "pred_action", "reason", "evidence_type")}
            for r in prop
            if r["gold_action"] == "ANSWER" and r["pred_action"] == "ABSTAIN"
        ][:15],
        "unnecessary_clarify": [
            {k: r[k] for k in ("example_id", "question", "gold_action", "pred_action", "reason", "evidence_type")}
            for r in prop
            if r["pred_action"] == "CLARIFY" and r["gold_action"] != "CLARIFY"
        ][:15],
        "retrieval_miss_on_answerable": [
            {k: r[k] for k in ("example_id", "question", "gold_passage_id", "retrieved_ids")}
            for r in prop
            if r["gold_action"] == "ANSWER" and r.get("gold_passage_id") not in r.get("retrieved_ids", [])
        ][:15],
    }
    save_json(out / "error_analysis.json", errors)

    # Demo traces
    save_json(out / "demo_traces.json", prop[:8])

    print("\nWrote metrics to", out / "test_metrics.json")
    print("Figures in", fig_dir)


if __name__ == "__main__":
    main()
