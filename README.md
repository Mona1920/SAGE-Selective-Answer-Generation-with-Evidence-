<<<<<<< HEAD
# Evidence-Calibrated Abstention in RAG

Academic prototype: retrieve science passages, assess evidence, then **ANSWER**, **CLARIFY**, or **ABSTAIN**.

## What is implemented here

| Step | Status |
|------|--------|
| 1. Dataset + literature notes | `GEMINI_PROMPT.md`, this README |
| 2. SciQ label construction | `src/dataset.py` |
| 3. TF-IDF RAG | `src/retrieve.py`, `src/generate.py` |
| 4. Evidence features | `src/evidence.py` |
| 5. Rule + calibrated 3-action policy | `src/policy.py` |
| 6–7. Baselines, metrics, plots | `scripts/run_experiment.py` |
| 8. CLI demo | `scripts/demo.py` |

**Retriever note:** The design called for sentence-transformers + FAISS. This environment has Python 3.14 without reliable PyTorch/FAISS wheels, so experiments use **TF-IDF cosine search**. That is a real sparse RAG baseline, not a fake embedding model.

## Setup

```bash
pip install datasets scikit-learn matplotlib pandas numpy pyyaml tqdm
python scripts/run_experiment.py
```

Results: `results/test_metrics.json`, `results/figures/`.

Do not treat confidence as certainty. Thresholds are tuned on validation only.

## Gemini

Paste `GEMINI_PROMPT.md` into Google Gemini to rebuild the Colab version with dense embeddings if a GPU runtime is available.
=======
# SAGE-Selective-Answer-Generation-with-Evidence-
Research paper
>>>>>>> origin/main
