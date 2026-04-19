from __future__ import annotations

from typing import Dict, List

from .graph import ask
from .retrieval import retrieve
from .graph import EMBEDDER, COLLECTION


BASELINE_QA = [
    {
        "question": "What should be checked in a contract review?",
        "ground_truth": "Identify parties, dates, payment, scope, confidentiality, termination, governing law, indemnity, liability, assignment, notices, and signatures.",
    },
    {
        "question": "What should I review in an NDA?",
        "ground_truth": "Check one-way or mutual scope, representatives, exclusions, purpose, duration, return duties, compelled disclosure, residuals, and injunctive relief.",
    },
    {
        "question": "What belongs in a litigation timeline?",
        "ground_truth": "Record event date, document name, court, parties, docket number, and neutral event description in chronological order.",
    },
    {
        "question": "Can the assistant give legal advice?",
        "ground_truth": "No. It can summarize and identify document information, but it must refer legal advice to a qualified attorney.",
    },
    {
        "question": "What fields belong in a due diligence index?",
        "ground_truth": "Category, document name, date, parties, status, key risk notes, and source citation.",
    },
]


def fallback_score(answer: str, ground_truth: str) -> float:
    answer_terms = {w.lower() for w in answer.split() if len(w) > 4}
    truth_terms = {w.lower().strip(".,") for w in ground_truth.split() if len(w) > 4}
    if not truth_terms:
        return 0.0
    return round(len(answer_terms & truth_terms) / len(truth_terms), 2)


def run_baseline() -> List[Dict[str, object]]:
    rows = []
    for item in BASELINE_QA:
        result = ask(item["question"], thread_id="eval")
        contexts, _ = retrieve(item["question"], EMBEDDER, COLLECTION)
        rows.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                "faithfulness": result.get("faithfulness", 0.0),
                "answer_relevancy_fallback": fallback_score(result["answer"], item["ground_truth"]),
            }
        )
    return rows


if __name__ == "__main__":
    for row in run_baseline():
        print(row)

