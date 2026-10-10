"""Evaluation charts saved under results/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_bars(summaries: dict, path: Path):
    names = list(summaries.keys())
    metrics = [
        "unsupported_answer_rate",
        "coverage",
        "selective_risk",
        "action_accuracy",
        "false_abstention_rate",
    ]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(metrics):
        vals = [summaries[n][m] for n in names]
        ax.bar(x + i * width, vals, width, label=m.replace("_", " "))
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline comparison on the held-out SciQ-derived test set")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_confusion(matrix, labels, title, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_reliability(xs, ys, ns, ece_val, path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    if xs:
        sizes = [20 + 4 * n for n in ns]
        ax.scatter(xs, ys, s=sizes, label="bins")
        ax.plot(xs, ys, "-o")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Action accuracy in bin")
    ax.set_title(f"Reliability diagram (ECE={ece_val:.3f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, path)


def plot_risk_coverage(curves: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (cov, risk) in curves.items():
        ax.plot(cov, risk, label=name)
    ax.set_xlabel("Coverage (fraction of questions ranked as most answerable)")
    ax.set_ylabel("Selective risk")
    ax.set_title("Coverage vs. risk")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, path)


def plot_ablation(rows: dict, path: Path):
    names = list(rows.keys())
    acc = [rows[n]["action_accuracy"] for n in names]
    uns = [rows[n]["unsupported_answer_rate"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, acc, 0.4, label="action accuracy")
    ax.bar(x + 0.2, uns, 0.4, label="unsupported-answer rate")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Feature ablations (proposed calibrated 3-action model)")
    ax.legend()
    _save(fig, path)
