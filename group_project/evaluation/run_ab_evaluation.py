"""Run a reproducible offline A/B retrieval evaluation.

The repository environment does not currently include ragas or a configured
Gemini SDK, so this runner uses the same deterministic extractive generator
for both configurations and documents token-overlap proxy metrics in RESULT.md.
It does not call an external API.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.task4_chunking_indexing import (
    EMBEDDING_MODEL,
    chunk_documents_by_type,
    load_documents,
)
from src.calibrate_fallback_threshold import calibrate_threshold
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
    relevant_chunks = sum(source_matches(result, sources) for result in results)
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
    chunk_count = len(chunk_documents_by_type(documents))

    # Calibrate on the same current dense index used by retrieval. This keeps
    # RESULT.md aligned with the committed corpus, chunker and embedding model.
    calibration = calibrate_threshold(
        [case["question"] for case in dataset]
    )

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
        "source_document_count": len(documents),
        "chunk_count": chunk_count,
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "threshold_calibration": calibration,
        "embedding_model": os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
        "generator": "deterministic extractive baseline, identical for A/B",
        "evaluator": "offline lexical/source proxies, no external API",
        "metric_definitions": {
            "faithfulness": "share of answer content tokens present in retrieved text",
            "answer_relevance": "share of question content tokens present in answer",
            "context_recall": "share of expected-answer content tokens present in retrieved text",
            "context_precision": "share of top-k chunks from expected source files",
        },
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

    report_lines = [
        "# Báo cáo đánh giá RAG",
        "",
        "## Run information / Thông tin lần chạy",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Evaluation date | {results['run_at']} |",
        "| Evaluator | Offline lexical/source proxies in `run_ab_evaluation.py` |",
        "| Generator | Deterministic extractive baseline (same for A/B) |",
        f"| Embedding model | {results['embedding_model']} |",
        f"| Code commit at evaluation time | {results['corpus_commit']} |",
        f"| Source documents | {results['source_document_count']} |",
        f"| Golden dataset size | {results['dataset_size']} |",
        f"| Indexed chunk count | {results['chunk_count']} |",
        f"| `top_k` | {results['top_k']} |",
        "| Chunk sizes (legal/news) | 1200/150 and 700/80 characters/overlap |",
        "",
        "## Evaluation method / Phương pháp",
        "",
        "- **Config A — dense-only:** semantic search over ChromaDB.",
        "- **Config B — hybrid + RRF:** semantic search and BM25, fused once with RRF.",
        "",
        f"Both configurations use the same corpus, chunks, {len(dataset)} golden questions, embedding model, `top_k`, extractive generator and evaluator.",
        "",
        "The four measures are deterministic proxies calculated from non-stopword token overlap and expected source filenames:",
        "",
        "- **Faithfulness proxy:** answer tokens also present in retrieved text / answer tokens.",
        "- **Answer relevance proxy:** question tokens present in answer / question tokens.",
        "- **Context recall proxy:** expected-answer tokens present in retrieved text / expected-answer tokens.",
        "- **Context precision proxy:** top-k chunks from a source file named in `expected_context` / top-k chunks.",
        "",
        "## Fallback threshold calibration",
        "",
        "Threshold selection uses dense top-1 cosine similarity. The calibration set contains "
        f"{calibration['in_domain_count']} in-domain and {calibration['out_of_domain_count']} fixed out-of-domain queries.",
        "",
        "| Configured threshold | Recommended threshold | Balanced accuracy | In-domain score range | Out-of-domain score range | False negatives | False positives |",
        "| ---: | ---: | ---: | --- | --- | ---: | ---: |",
        f"| {results['score_threshold']:.4f} | {calibration['recommended_SCORE_THRESHOLD']:.4f} | {calibration['balanced_accuracy']:.3f} | {calibration['in_domain_score_range']} | {calibration['out_of_domain_score_range']} | {len(calibration['false_negative_in_domain_queries'])} | {len(calibration['false_positive_out_of_domain_queries'])} |",
        "",
        "## Overall scores",
        "",
        "| Metric | Config A | Config B | Delta B−A |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "faithfulness": "Faithfulness proxy",
        "answer_relevance": "Answer relevance proxy",
        "context_recall": "Context recall proxy",
        "context_precision": "Context precision proxy",
    }
    for key, label in labels.items():
        report_lines.append(
            f"| {label} | {fmt(dense_metrics[key])} | {fmt(hybrid_metrics[key])} | {fmt(deltas[key])} |"
        )
    report_lines.extend(
        [
            "## A/B comparison",
            "",
            f"| Configuration | Total time for {len(dataset)} queries | Time per query |",
            "| --- | ---: | ---: |",
            f"| Dense-only | {dense_seconds:.2f}s | {dense_seconds / len(dataset):.3f}s |",
            f"| Hybrid + RRF | {hybrid_seconds:.2f}s | {hybrid_seconds / len(dataset):.3f}s |",
            "",
            "Metric deltas are interpreted separately. No aggregate score is used to claim that one retriever is universally better.",
            "",
            "## Worst performers",
            "",
            "The table lists the three cases with the lowest context-recall proxy; it reports measured values without assigning an unverified cause.",
            "",
            "| Rank | Question | Configuration | Answer relevance proxy | Context recall proxy | Context precision proxy |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    evaluated = []
    for config, rows in (("Dense-only", dense_rows), ("Hybrid + RRF", hybrid_rows)):
        for row in rows:
            evaluated.append(
                (row["context_recall"], row["answer_relevance"], config, row)
            )
    evaluated.sort(key=lambda item: (item[0], item[1]))
    for rank, (recall, _, config, row) in enumerate(evaluated[:3], 1):
        report_lines.append(
            f"| {rank} | {row['question']} | {config} | {fmt(row['answer_relevance'])} | {fmt(recall)} | {fmt(row['context_precision'])} |"
        )
    report_lines.extend(
        [
            "",
            "## Limitations and interpretation",
            "",
            "- The four measures are deterministic lexical/source proxies, not semantic judgments from Ragas or a human evaluator.",
            "- Answers are extracted directly from retrieved chunks; the faithfulness proxy is therefore expected to be high and does not independently establish semantic correctness.",
            "- Context precision is measured by expected source filename, not by expert relevance labels for each chunk.",
            "- Threshold separation is measured on 16 in-domain and 8 fixed out-of-domain queries; this sample does not establish performance on unseen queries.",
            "- Latency is one warmed local run over the golden set, not a repeated benchmark or production estimate.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "deltas": deltas, "chunks": chunk_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
