"""Estimate a dense-score threshold from in-domain and out-of-domain queries.

The in-domain examples come from the golden dataset. Out-of-domain examples
are kept separately so the threshold can be recalibrated when the corpus or
embedding model changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "group_project" / "evaluation"
GOLDEN_DATASET = EVALUATION_DIR / "golden_dataset.json"
OUT_OF_DOMAIN_DATASET = EVALUATION_DIR / "out_of_domain_queries.json"
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.task5_semantic_search import semantic_search  # noqa: E402


def _best_dense_score(question: str) -> float:
    results = semantic_search(question, top_k=1)
    return float(results[0]["score"]) if results else 0.0


def _balanced_accuracy(
    in_domain_scores: list[float],
    out_of_domain_scores: list[float],
    threshold: float,
) -> float:
    true_positive_rate = sum(score >= threshold for score in in_domain_scores) / len(
        in_domain_scores
    )
    true_negative_rate = sum(score < threshold for score in out_of_domain_scores) / len(
        out_of_domain_scores
    )
    return (true_positive_rate + true_negative_rate) / 2


def recommend_threshold(
    in_domain_scores: list[float], out_of_domain_scores: list[float]
) -> tuple[float, float]:
    """Return the midpoint when separable, otherwise maximize balanced accuracy."""
    if not in_domain_scores or not out_of_domain_scores:
        raise ValueError("Both query sets must contain at least one score")

    lowest_in_domain = min(in_domain_scores)
    highest_out_of_domain = max(out_of_domain_scores)
    if lowest_in_domain > highest_out_of_domain:
        threshold = (lowest_in_domain + highest_out_of_domain) / 2
        return threshold, 1.0

    unique_scores = sorted(set(in_domain_scores + out_of_domain_scores))
    candidates = {0.0, 0.5, 1.0}
    candidates.update(
        (left + right) / 2 for left, right in zip(unique_scores, unique_scores[1:])
    )
    threshold = min(
        candidates,
        key=lambda candidate: (
            -_balanced_accuracy(in_domain_scores, out_of_domain_scores, candidate),
            abs(candidate - 0.5),
            candidate,
        ),
    )
    return threshold, _balanced_accuracy(
        in_domain_scores, out_of_domain_scores, threshold
    )


def main() -> None:
    golden = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    out_of_domain = json.loads(OUT_OF_DOMAIN_DATASET.read_text(encoding="utf-8"))
    in_domain_questions = [
        item["question"]
        for item in golden
        if isinstance(item, dict) and isinstance(item.get("question"), str)
    ]
    out_of_domain_questions = [
        item
        for item in out_of_domain
        if isinstance(item, str) and item.strip()
    ]
    if not in_domain_questions or not out_of_domain_questions:
        raise ValueError("Calibration query sets must both be non-empty")

    in_domain_scores = [
        _best_dense_score(question) for question in in_domain_questions
    ]
    out_of_domain_scores = [
        _best_dense_score(question) for question in out_of_domain_questions
    ]
    threshold, balanced_accuracy = recommend_threshold(
        in_domain_scores, out_of_domain_scores
    )

    print(
        json.dumps(
            {
                "recommended_SCORE_THRESHOLD": round(threshold, 4),
                "balanced_accuracy": round(balanced_accuracy, 4),
                "in_domain_count": len(in_domain_scores),
                "out_of_domain_count": len(out_of_domain_scores),
                "in_domain_score_range": [
                    round(min(in_domain_scores), 4),
                    round(max(in_domain_scores), 4),
                ],
                "out_of_domain_score_range": [
                    round(min(out_of_domain_scores), 4),
                    round(max(out_of_domain_scores), 4),
                ],
                "false_negative_in_domain_queries": [
                    question
                    for question, score in zip(in_domain_questions, in_domain_scores)
                    if score < threshold
                ],
                "false_positive_out_of_domain_queries": [
                    question
                    for question, score in zip(
                        out_of_domain_questions, out_of_domain_scores
                    )
                    if score >= threshold
                ],
                "instruction": "Copy recommended_SCORE_THRESHOLD to .env and review false positives/negatives.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
