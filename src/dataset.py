"""
Step 2: load SciQ and construct ANSWER / CLARIFY / ABSTAIN examples.

SciQ does not contain these labels. We create them with documented rules.
Gold action labels are used only for training the decision model and for
evaluation. They are never shown to the retriever or generator.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Literal

from datasets import load_dataset

from .utils import normalize_text

Action = Literal["ANSWER", "CLARIFY", "ABSTAIN"]
EvidenceType = Literal["sufficient", "missing", "contradictory", "ambiguous"]


@dataclass
class Passage:
    passage_id: str
    text: str
    source: str


@dataclass
class Example:
    example_id: str
    split: str
    question: str
    options: list[str]
    gold_answer: str
    gold_action: Action
    evidence_type: EvidenceType
    gold_passage_id: str | None
    withhold_passage_ids: list[str]
    injected_conflict_passage: str | None
    construction_note: str
    clarification_target: str | None


AMBIGUOUS_TEMPLATES = [
    "What is this called?",
    "Which one is correct?",
    "What happens in this case?",
    "What is the answer to this?",
    "Can you tell me which process this is?",
]


def load_sciq():
    """Download the official SciQ splits from Hugging Face."""
    return load_dataset("allenai/sciq")


def _options(row) -> list[str]:
    return [
        row["correct_answer"],
        row["distractor1"],
        row["distractor2"],
        row["distractor3"],
    ]


def build_corpus(sciq) -> list[Passage]:
    """
    Retrieval corpus = unique non-empty SciQ support paragraphs.

    We keep train+validation+test supports because this is a closed science
    collection, not an open-web search engine. Gold-passage withholding is
    how we simulate missing evidence, not by hiding the whole test set.
    """
    seen = {}
    passages = []
    for split in ("train", "validation", "test"):
        for i, row in enumerate(sciq[split]):
            text = (row.get("support") or "").strip()
            if not text:
                continue
            key = normalize_text(text)
            if key in seen:
                continue
            pid = f"{split}_{i}"
            seen[key] = pid
            passages.append(
                Passage(
                    passage_id=pid,
                    text=text,
                    source=f"sciq:{split}:{i}",
                )
            )
    return passages


def _find_passage_id(row, split: str, idx: int, text_to_id: dict[str, str]) -> str | None:
    text = (row.get("support") or "").strip()
    if not text:
        return None
    return text_to_id.get(normalize_text(text), f"{split}_{idx}")


def _underspecify_question(question: str, gold_answer: str) -> str:
    """
    Remove answer words and specific content nouns so the question cannot
    be answered without asking what the user meant.
    """
    q = question.strip()
    answer_words = set(normalize_text(gold_answer).split())
    tokens = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", q)
    kept = []
    for tok in tokens:
        if tok.lower() in answer_words and tok.isalpha() and len(tok) > 3:
            continue
        kept.append(tok)
    reduced = "".join(
        (" " + t) if t.isalnum() or t.startswith("'") else t for t in kept
    ).strip()
    reduced = re.sub(r"\s+", " ", reduced)
    if len(reduced.split()) < 5:
        return random.choice(AMBIGUOUS_TEMPLATES)
    # Replace a long noun-ish span with a pronoun.
    reduced = re.sub(
        r"\b(the|a|an)\s+[a-z]{5,}(\s+[a-z]{5,}){0,3}\b",
        " this ",
        reduced,
        count=1,
        flags=re.I,
    )
    reduced = re.sub(r"\s+", " ", reduced).strip()
    if reduced.lower() == normalize_text(question) or len(reduced.split()) < 5:
        return random.choice(AMBIGUOUS_TEMPLATES)
    return reduced


def _conflict_passage(gold: str, distractor: str) -> str:
    return (
        f"Several sources incorrectly mention {gold}. "
        f"The scientifically correct term is {distractor}. "
        f"Do not use {gold}; {distractor} is the accepted answer in this document."
    )


def construct_split(
    sciq_split,
    split_name: str,
    n_per_class: int,
    text_to_id: dict[str, str],
    rng: random.Random,
) -> list[Example]:
    """
    Create a balanced evaluation/training slice.

    ANSWER: original question, gold passage remains retrievable.
    ABSTAIN/missing: gold passage withheld (or originally empty support).
    ABSTAIN/contradictory: extra passage asserts a distractor.
    CLARIFY: underspecified question; user must specify the referent.
    """
    rows = list(enumerate(sciq_split))
    with_support = [(i, r) for i, r in rows if (r.get("support") or "").strip()]
    without_support = [(i, r) for i, r in rows if not (r.get("support") or "").strip()]
    rng.shuffle(with_support)
    rng.shuffle(without_support)

    examples: list[Example] = []
    used_ids: set[int] = set()

    def take_supported(n: int) -> list[tuple[int, dict]]:
        picked = []
        rest = []
        for item in with_support:
            if item[0] in used_ids:
                continue
            if len(picked) < n:
                picked.append(item)
                used_ids.add(item[0])
            else:
                rest.append(item)
        if len(picked) < n:
            raise RuntimeError(f"{split_name}: need {n} supported rows, found {len(picked)}")
        return picked

    def take_missing(n: int) -> list[tuple[int, dict]]:
        picked = []
        for item in without_support + with_support:
            if item[0] in used_ids:
                continue
            picked.append(item)
            used_ids.add(item[0])
            if len(picked) == n:
                break
        if len(picked) < n:
            raise RuntimeError(f"{split_name}: need {n} missing-evidence rows, found {len(picked)}")
        return picked

    # ANSWER
    for i, row in take_supported(n_per_class):
        pid = _find_passage_id(row, split_name, i, text_to_id)
        examples.append(
            Example(
                example_id=f"{split_name}-answer-{i}",
                split=split_name,
                question=row["question"].strip(),
                options=_options(row),
                gold_answer=row["correct_answer"].strip(),
                gold_action="ANSWER",
                evidence_type="sufficient",
                gold_passage_id=pid,
                withhold_passage_ids=[],
                injected_conflict_passage=None,
                construction_note="Original SciQ question; gold support remains in the corpus.",
                clarification_target=None,
            )
        )

    # ABSTAIN missing evidence
    for i, row in take_missing(n_per_class):
        pid = _find_passage_id(row, split_name, i, text_to_id)
        withhold = [pid] if pid else []
        examples.append(
            Example(
                example_id=f"{split_name}-missing-{i}",
                split=split_name,
                question=row["question"].strip(),
                options=_options(row),
                gold_answer=row["correct_answer"].strip(),
                gold_action="ABSTAIN",
                evidence_type="missing",
                gold_passage_id=pid,
                withhold_passage_ids=withhold,
                injected_conflict_passage=None,
                construction_note=(
                    "Gold support withheld or originally empty, so retrieved evidence should be insufficient."
                ),
                clarification_target=None,
            )
        )

    # ABSTAIN contradictory
    for i, row in take_supported(n_per_class):
        pid = _find_passage_id(row, split_name, i, text_to_id)
        distractor = row["distractor1"].strip()
        examples.append(
            Example(
                example_id=f"{split_name}-conflict-{i}",
                split=split_name,
                question=row["question"].strip(),
                options=_options(row),
                gold_answer=row["correct_answer"].strip(),
                gold_action="ABSTAIN",
                evidence_type="contradictory",
                gold_passage_id=pid,
                withhold_passage_ids=[],
                injected_conflict_passage=_conflict_passage(row["correct_answer"].strip(), distractor),
                construction_note="Injected a passage that asserts a distractor and denies the gold answer.",
                clarification_target=None,
            )
        )

    # CLARIFY
    for i, row in take_supported(n_per_class):
        pid = _find_passage_id(row, split_name, i, text_to_id)
        ambiguous_q = _underspecify_question(row["question"], row["correct_answer"])
        examples.append(
            Example(
                example_id=f"{split_name}-clarify-{i}",
                split=split_name,
                question=ambiguous_q,
                options=_options(row),
                gold_answer=row["correct_answer"].strip(),
                gold_action="CLARIFY",
                evidence_type="ambiguous",
                gold_passage_id=pid,
                withhold_passage_ids=[],
                injected_conflict_passage=None,
                construction_note=(
                    f"Question underspecified from original SciQ item. Original: {row['question']}"
                ),
                clarification_target=row["question"].strip(),
            )
        )

    rng.shuffle(examples)
    return examples


def construct_all(sciq, n_per_class: dict, seed: int) -> tuple[list[Passage], dict[str, list[Example]]]:
    rng = random.Random(seed)
    passages = build_corpus(sciq)
    text_to_id = {normalize_text(p.text): p.passage_id for p in passages}
    data = {}
    for split in ("train", "validation", "test"):
        data[split] = construct_split(
            sciq[split],
            split,
            n_per_class[split],
            text_to_id,
            rng,
        )
    return passages, data


def examples_to_json(examples: list[Example]) -> list[dict]:
    return [asdict(e) for e in examples]
