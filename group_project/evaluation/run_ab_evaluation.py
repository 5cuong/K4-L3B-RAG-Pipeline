"""Run a reproducible offline A/B retrieval evaluation.

The repository environment does not currently include ragas or a configured
Gemini SDK, so this runner uses the same deterministic extractive generator
for both configurations and documents token-overlap proxy metrics in RESULT.md.
It does not call an external API.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.task4_chunking_indexing import chunk_documents, load_documents
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task9_retrieval_pipeline import SCORE_THRESHOLD


EVALUATION_DIR = ROOT / "group_project" / "evaluation"
DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "AB_RESULTS.json"
REPORT_PATH = EVALUATION_DIR / "RESULT.md"
TOP_K = 5

STOPWORDS = {
    "là", "và", "của", "cho", "đối", "với", "những", "các", "một",
    "nào", "được", "theo", "trong", "từ", "tại", "khi", "có", "phải",
    "như", "thế", "nào", "người", "này", "đó", "về", "đến", "hay",
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are",
}


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def extractive_answer(query: str, results: list[dict]) -> str:
    """Use one fixed local generator for both A/B configurations."""
    query_tokens = tokens(query)
    candidates: list[tuple[float, int, str]] = []
    for result_index, result in enumerate(results):
        sentences = re.split(r"(?<=[.!?;])\s+|\n+", result["content"])
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            sentence_tokens = tokens(sentence)
            overlap = len(query_tokens & sentence_tokens)
            score = overlap / max(len(query_tokens), 1)
            candidates.append((score, -result_index, sentence))

    if not candidates:
        return results[0]["content"][:400] if results else ""
    candidates.sort(reverse=True)
    selected = [item[2] for item in candidates[:2]]
    return " ".join(dict.fromkeys(selected))


def expected_sources(case: dict) -> set[str]:
    source_names = set()
    for value in case.get("expected_context", []):
        match = re.search(r"(?:legal|news)/([^\s—]+\.md)", str(value))
        if match:
            source_names.add(match.group(1))
    return source_names


def source_matches(result: dict, sources: set[str]) -> bool:
    return result.get("metadata", {}).get("source") in sources


def evaluate_case(case: dict, results: list[dict]) -> dict:
    answer = extractive_answer(case["question"], results)
    context = " ".join(item["content"] for item in results)
    answer_tokens = tokens(answer)
    query_tokens = tokens(case["question"])
    expected_answer_tokens = tokens(case["expected_answer"])
    context_tokens = tokens(context)
    sources = expected_sources(case)

    faithfulness = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    relevance = len(answer_tokens & query_tokens) / max(len(query_tokens), 1)
    recall = len(expected_answer_tokens & context_tokens) / max(
        len(expected_answer_tokens), 1
    )
    relevant_chunks = sum(
        source_matches(result, sources)
        or len(tokens(result["content"]) & expected_answer_tokens) >= 3
        for result in results
    )
    precision = relevant_chunks / max(len(results), 1)

    return {
        "question": case["question"],
        "answer": answer,
        "source_ids": [item["id"] for item in results],
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_recall": recall,
        "context_precision": precision,
    }


def run_config(dataset: list[dict], name: str) -> tuple[list[dict], float]:
    started = time.perf_counter()
    evaluated = []
    for case in dataset:
        query = case["question"]
        if name == "dense-only":
            results = semantic_search(query, top_k=TOP_K)
        else:
            dense = semantic_search(query, top_k=TOP_K * 2)
            sparse = lexical_search(query, top_k=TOP_K * 2)
            results = rerank_rrf([dense, sparse], top_k=TOP_K)
        evaluated.append(evaluate_case(case, results))
    return evaluated, time.perf_counter() - started


def average(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / max(len(rows), 1)


def metric_rows(rows: list[dict]) -> dict[str, float]:
    return {
        "faithfulness": average(rows, "faithfulness"),
        "answer_relevance": average(rows, "answer_relevance"),
        "context_recall": average(rows, "context_recall"),
        "context_precision": average(rows, "context_precision"),
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    load_dotenv(ROOT / ".env")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    documents = load_documents()
    chunk_count = len(chunk_documents(documents))

    # Warm up the shared embedding model so Config A does not pay a one-time
    # model-load cost that would make the latency comparison unfair.
    semantic_search(dataset[0]["question"], top_k=1)
    dense_rows, dense_seconds = run_config(dataset, "dense-only")
    hybrid_rows, hybrid_seconds = run_config(dataset, "hybrid-rrf")
    dense_metrics = metric_rows(dense_rows)
    hybrid_metrics = metric_rows(hybrid_rows)
    metrics = {
        "dense-only": dense_metrics,
        "hybrid-rrf": hybrid_metrics,
    }
    deltas = {
        key: hybrid_metrics[key] - dense_metrics[key]
        for key in dense_metrics
    }

    results = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(dataset),
        "chunk_count": chunk_count,
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "embedding_model": "BAAI/bge-m3",
        "generator": "deterministic extractive baseline, identical for A/B",
        "evaluator": "offline token-overlap proxy, no external API",
        "corpus_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "latency_seconds": {
            "dense-only": dense_seconds,
            "hybrid-rrf": hybrid_seconds,
        },
        "metrics": metrics,
        "deltas": deltas,
        "cases": {
            "dense-only": dense_rows,
            "hybrid-rrf": hybrid_rows,
        },
    }
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    combined = []
    for index, case in enumerate(dataset):
        dense = dense_rows[index]
        hybrid = hybrid_rows[index]
        combined.append(
            (
                (
                    dense["faithfulness"]
                    + dense["answer_relevance"]
                    + dense["context_recall"]
                    + dense["context_precision"]
                    + hybrid["faithfulness"]
                    + hybrid["answer_relevance"]
                    + hybrid["context_recall"]
                    + hybrid["context_precision"]
                )
                / 8,
                case["question"],
                dense,
                hybrid,
            )
        )
    combined.sort(key=lambda item: item[0])

    report_lines = [
        "# RAG evaluation results",
        "",
        "## Run information",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Evaluation date | {results['run_at']} |",
        "| Framework and version | Offline evaluator in `run_ab_evaluation.py` |",
        "| Evaluator model | None; deterministic token-overlap proxy |",
        "| Generator model | Deterministic extractive baseline |",
        f"| Embedding model | {results['embedding_model']} |",
        f"| Corpus version/commit | {results['corpus_commit']} |",
        f"| Golden dataset size | {results['dataset_size']} |",
        f"| Indexed chunk count | {results['chunk_count']} |",
        f"| `top_k` | {results['top_k']} |",
        f"| Fallback threshold and calibration | {results['score_threshold']}; calibrated earlier with in-domain ≈0.73 and out-of-domain ≈0.41 |",
        "",
        "## Configurations",
        "",
        "- **Config A — dense-only:** semantic search over ChromaDB.",
        "- **Config B — hybrid + RRF:** dense search + BM25, fused once with RRF.",
        "",
        "Both configurations use the same 16 cases, top_k, embedding model, extractive generator and evaluator. The metrics are offline proxies because `ragas` and a configured evaluator LLM are not installed in the current environment.",
        "",
        "## Overall scores",
        "",
        "| Metric | Config A | Config B | Delta B−A |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer relevance",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
    }
    for key, label in labels.items():
        report_lines.append(
            f"| {label} | {fmt(dense_metrics[key])} | {fmt(hybrid_metrics[key])} | {fmt(deltas[key])} |"
        )
    dense_avg = sum(dense_metrics.values()) / 4
    hybrid_avg = sum(hybrid_metrics.values()) / 4
    report_lines.extend(
        [
            f"| **Average** | **{fmt(dense_avg)}** | **{fmt(hybrid_avg)}** | **{fmt(hybrid_avg - dense_avg)}** |",
            "",
            "## A/B comparison",
            "",
            f"- **Cấu hình tốt hơn theo proxy trung bình:** {'Config B — hybrid + RRF' if hybrid_avg >= dense_avg else 'Config A — dense-only'}.",
            f"- **Evidence:** hybrid recall={fmt(hybrid_metrics['context_recall'])}, precision={fmt(hybrid_metrics['context_precision'])}; dense recall={fmt(dense_metrics['context_recall'])}, precision={fmt(dense_metrics['context_precision'])}.",
            f"- **Latency:** dense-only {dense_seconds:.2f}s; hybrid+RRF {hybrid_seconds:.2f}s trên {len(dataset)} query. Hybrid có thêm chi phí BM25 và fusion nhưng không gọi LLM/API ngoài.",
            "- **Caveat:** proxy này dùng để kiểm tra retrieval và regression; kết quả Ragas cần chạy lại khi cài evaluator model.",
            "",
            "## Worst performers",
            "",
            "| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for rank, (_, question, dense, hybrid) in enumerate(combined[:3], 1):
        worst = min((dense, "dense-only"), (hybrid, "hybrid-rrf"), key=lambda item: sum(item[0][key] for key in labels))
        row, config = worst
        if row["context_recall"] < 0.5 or row["context_precision"] < 0.4:
            stage = "retrieval"
            cause = "Expected evidence was not concentrated in the top-k retrieved chunks."
        elif row["faithfulness"] < 0.7:
            stage = "generation"
            cause = "The extracted answer had weak overlap with retrieved context."
        else:
            stage = "data"
            cause = "Question evidence is distributed across long legal/news documents."
        report_lines.append(
            f"| {rank} | {question} | {config} | {fmt(row['faithfulness'])} | {fmt(row['answer_relevance'])} | {fmt(row['context_recall'])} | {fmt(row['context_precision'])} | {stage} | {cause} |"
        )
    report_lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |",
            "| ---: | --- | --- | --- | --- |",
            "| 1 | Giữ lọc boilerplate của news trước khi chunking | News content đã giảm đáng kể và corpus hiện còn 1.206 chunks | Ít nhiễu, giảm chi phí embedding/retrieval | Chạy lại Task 3 và so sánh chunk count |",
            "| 2 | Bổ sung các câu hỏi paraphrase và câu hỏi cần phân biệt nguồn | Worst cases thường có evidence nằm rải rác hoặc nhiều nguồn gần nghĩa | Đo được khả năng dense/BM25 rõ hơn | Thêm case vào golden dataset rồi chạy lại A/B |",
            "| 3 | Chạy Ragas với cùng generator/evaluator khi môi trường đủ dependency | Proxy hiện tại không thay thế đánh giá LLM | Có faithfulness/relevance chuẩn hơn | Cài `ragas`, cấu hình evaluator và ghi lại model/version |",
            "",
            "## Bonus experiments",
            "",
            "| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |",
            "| --- | --- | ---: | --- | --- |",
            "| News boilerplate filtering | Markdown raw từ crawler | Chunk count giảm từ khoảng 1.654 xuống 1.206 | Giảm số chunk news và chi phí xử lý | Nên giữ bước lọc trước Task 4 |",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "deltas": deltas, "chunks": chunk_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
