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
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.task5_semantic_search import semantic_search  # noqa: E402


# Keep the small OOD calibration set with the calibration code instead of
# requiring a separate submission data artifact.
OUT_OF_DOMAIN_QUESTIONS = [
    "Dự báo thời tiết ở Hà Nội ngày mai như thế nào?",
    "Cách nấu phở bò tại nhà gồm những bước nào?",
    "Viết chương trình Python sắp xếp một danh sách số nguyên.",
    "Lịch thi đấu bóng đá Việt Nam cuối tuần này ra sao?",
    "Thuế thu nhập cá nhân được tính như thế nào?",
    "Thủ tục đăng ký xe máy mới cần những giấy tờ gì?",
    "Quy định xin visa du lịch Nhật Bản gồm những điều kiện nào?",
    "Cách chăm sóc cây lan khi lá bị vàng?",
]


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


def calibrate_threshold(
    in_domain_questions: list[str] | None = None,
    out_of_domain_questions: list[str] | None = None,
) -> dict:
    """Score both query sets and return threshold metrics and error examples."""
    golden = json.loads(GOLDEN_DATASET.read_text(encoding="utf-8"))
    if in_domain_questions is None:
        in_domain_questions = [
            item["question"]
            for item in golden
            if isinstance(item, dict) and isinstance(item.get("question"), str)
        ]
    if out_of_domain_questions is None:
        out_of_domain_questions = OUT_OF_DOMAIN_QUESTIONS
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

    return {
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
            for question, score in zip(out_of_domain_questions, out_of_domain_scores)
            if score >= threshold
        ],
    }


def main() -> None:
    print(json.dumps(calibrate_threshold(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
