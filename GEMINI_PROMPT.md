# Gemini prompt: Evidence-Calibrated Abstention in RAG

Copy everything below the line into Google Gemini (prefer Gemini 2.5 Pro / a long-context model). Work in Google Colab. Do not skip the integrity rules.

---

You are an AI/ML research engineer, Python developer, and research mentor.

Build a complete, working academic project titled:

**“Evidence-Calibrated Abstention in Retrieval-Augmented Generation (RAG): Answer, Clarify, or Abstain.”**

## Objective

Build a RAG system that retrieves documents, assesses whether the evidence supports an answer, then chooses one action:

1. ANSWER — supported answer with citations
2. CLARIFY — ask a targeted question when the user request is ambiguous
3. ABSTAIN — say the evidence is insufficient or contradictory and explain why

Research goal: reduce unsupported answers while keeping useful coverage.

This is an M.Tech CSE research project: working code, reproducible experiments, baselines, plots, error analysis, and a paper. It is not a chatbot demo.

## Environment

- Python, Google Colab, free CPU/GPU
- Beginner-friendly libraries
- No training LLMs from scratch
- No paid APIs or API keys for the initial implementation
- Prefer Hugging Face `allenai/sciq`, `sentence-transformers`, FAISS if available, sklearn, matplotlib, Gradio
- If sentence-transformers/FAISS fail, use TF-IDF + cosine search and document the limitation
- Explain important lines in simple language
- Modular, reproducible code

## Integrity (non-negotiable)

- Do not invent citations, accuracy numbers, charts, or findings
- Do not claim novelty until closest papers are reviewed
- Tune thresholds on validation only; never on test labels
- A weighted score is not a calibrated probability
- Semantic similarity is not proof of evidential support
- SciQ does **not** already contain ANSWER/CLARIFY/ABSTAIN labels; you must construct and document them
- If the proposed method loses to a baseline, report that honestly
- Distinguish proposed ideas, implemented methods, and experimentally verified findings

## Verified starting literature (do not fabricate extras)

Read/cite these if you mention related work:

- Welbl, Liu, Gardner, 2017. SciQ. ACL WNUT. https://aclanthology.org/W17-4413/
- Lewis et al., 2020. RAG. NeurIPS. https://arxiv.org/abs/2005.11401
- Rajpurkar et al., 2018. SQuAD 2.0
- Min et al., 2020. AmbigQA. EMNLP
- Kamath, Jia, Liang, 2020. Selective Question Answering under Domain Shift. ACL
- Guo et al., 2017. On Calibration of Modern Neural Networks. ICML
- Chen et al., 2024. RGB benchmark. AAAI / https://arxiv.org/abs/2309.01431
- Asai et al., 2024. Self-RAG. ICLR. https://arxiv.org/abs/2310.11511
- Yan et al., 2024. CRAG. https://arxiv.org/abs/2401.15884
- Kuhn et al., 2022. CLAM. https://arxiv.org/abs/2212.07769
- Cole et al., 2023. Selectively Answering Ambiguous Questions. EMNLP
- Kim et al., 2023. Tree of Clarifications. EMNLP
- Joren et al., 2025. Sufficient Context. ICLR / https://arxiv.org/pdf/2411.06037
- Zhang and Wu, 2026. Evidence Sufficiency Benchmark. CMC https://www.techscience.com/cmc/v89n1/68467
- Dasigi et al., 2021. QASPER. https://arxiv.org/abs/2105.03011

Closest prior work already studies RAG abstention and sufficiency. The candidate (unverified) gap is whether a **lightweight three-action** policy (ANSWER/CLARIFY/ABSTAIN) with **evidence features + calibration** helps on a small science corpus versus always-answer RAG and binary abstention. Do not claim this is novel.

## Dataset recommendation

Start with SciQ (`load_dataset("allenai/sciq")`).

- 13,679 science MCQs; train 11679 / val 1000 / test 1000
- Fields: question, distractor1/2/3, correct_answer, support
- Support missing for some rows (unreleasable texts; direct-answer subset 10481/887/884 in the paper)
- Later optional second source: QASPER, not raw PDFs first

## Exact roadmap — one step at a time

Do not dump the whole project in one notebook cell. After each step, print expected outputs and wait for the user if a step fails.

### STEP 1 — Collect research documents

Dataset comparison, SciQ recommendation, literature table, unverified gap, RQ hypotheses.

### STEP 2 — Prepare the dataset

Inspect columns. Deduplicate. Prevent split leakage.

Construct a documented evaluation set with gold actions:

- ANSWER: original question, gold support stays in the corpus
- ABSTAIN/missing: empty support or withhold gold passage from retrieval
- ABSTAIN/contradictory: inject a passage that asserts a distractor
- CLARIFY: underspecify the question (pronouns, missing entity)

Save label definitions. Do not leak construction flags (e.g. “injected_conflict”) into model features.

### STEP 3 — Basic RAG baseline

question → retrieve top-k → generate only from context → citations (passage ids).

Establish always-answer RAG before the proposed method.

### STEP 4 — Evidence assessment

Features: retrieval relevance, answer mention / sufficiency, option margin, contradiction among options/passages, question ambiguity, empty retrieval.

Compare rules vs NLI if Colab can load a small MNLI model. Create unit test cases.

### STEP 5 — Decision policy

IF question materially ambiguous: CLARIFY  
ELSE IF evidence sufficiently supports one answer: ANSWER with citations  
ELSE: ABSTAIN and explain  

Start with transparent rules, then logistic regression, then calibration (Platt / CalibratedClassifierCV). Tune thresholds on validation. Log action, confidence, passages, reason.

### STEP 6 — Baselines (same test questions, corpus, generator)

1. Standard RAG, always answer
2. Fixed retrieval/confidence threshold
3. Calibrated binary answer-vs-abstain
4. Proposed three-action system

Ablate individual features.

### STEP 7 — Evaluate

Metrics (define each in plain language):

- Exact match / option accuracy
- Unsupported-answer rate
- False abstention rate
- Coverage
- Selective risk vs coverage
- Brier, ECE
- Clarification precision/recall
- Citation hit rate
- Latency

Plots: coverage-vs-risk, reliability diagram, confusion matrices, ablation bars.

Error analysis: unsupported answers, unnecessary abstention, unnecessary clarification, retrieval misses, conflicts.

Never invent numbers.

### STEP 8 — Demo + paper

Gradio (or CLI) showing question, retrieved evidence, features, action, response, calibrated probability (not certainty), latency.

Paper sections: title, abstract, intro, related work, gap/RQs, method, data/setup, baselines, results, ablations/errors, limitations, verified references.

## Research questions → hypotheses

- RQ1: evidence-based decisions reduce unsupported answers vs always-answer RAG
- RQ2: calibration improves coverage–risk vs a fixed uncalibrated threshold
- RQ3: CLARIFY beats binary answer/abstain on ambiguous questions
- RQ4: sufficiency/conflict features beat similarity alone

Evaluate with the metrics above.

## Teaching style

- Purpose of each step before code
- Complete Colab cells in order
- Expected output
- Common errors and fixes
- Working baseline before complexity

Begin with STEP 1, then implement STEP 2 in Colab, then continue through STEP 8 until metrics and plots exist. Save all results to disk. After experiments, write the paper using only those results.

---
