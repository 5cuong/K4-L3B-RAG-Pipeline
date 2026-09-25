# Báo cáo đánh giá RAG

## Run information / Thông tin lần chạy

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-25T08:34:58.265708+00:00 |
| Evaluator | Offline lexical/source proxies in `run_ab_evaluation.py` |
| Generator | Deterministic extractive baseline (same for A/B) |
| Embedding model | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Code commit at evaluation time | 5e3bb1d |
| Source documents | 9 |
| Golden dataset size | 16 |
| Indexed chunk count | 570 |
| `top_k` | 5 |
| Chunk sizes (legal/news) | 1200/150 and 700/80 characters/overlap |

## Evaluation method / Phương pháp

- **Config A — dense-only:** semantic search over ChromaDB.
- **Config B — hybrid + RRF:** semantic search and BM25, fused once with RRF.

Both configurations use the same corpus, chunks, 16 golden questions, embedding model, `top_k`, extractive generator and evaluator.

The four measures are deterministic proxies calculated from non-stopword token overlap and expected source filenames:

- **Faithfulness proxy:** answer tokens also present in retrieved text / answer tokens.
- **Answer relevance proxy:** question tokens present in answer / question tokens.
- **Context recall proxy:** expected-answer tokens present in retrieved text / expected-answer tokens.
- **Context precision proxy:** top-k chunks from a source file named in `expected_context` / top-k chunks.

## Fallback threshold calibration

Threshold selection uses dense top-1 cosine similarity. The calibration set contains 16 in-domain and 8 fixed out-of-domain queries.

| Configured threshold | Recommended threshold | Balanced accuracy | In-domain score range | Out-of-domain score range | False negatives | False positives |
| ---: | ---: | ---: | --- | --- | ---: | ---: |
| 0.5474 | 0.5474 | 1.000 | [0.6001, 0.8219] | [0.1901, 0.4947] | 0 | 0 |

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness proxy | 1.000 | 1.000 | 0.000 |
| Answer relevance proxy | 0.732 | 0.753 | 0.020 |
| Context recall proxy | 0.810 | 0.854 | 0.044 |
| Context precision proxy | 0.512 | 0.487 | -0.025 |
## A/B comparison

| Configuration | Total time for 16 queries | Time per query |
| --- | ---: | ---: |
| Dense-only | 1.04s | 0.065s |
| Hybrid + RRF | 2.66s | 0.166s |

Metric deltas are interpreted separately. No aggregate score is used to claim that one retriever is universally better.

## Worst performers

The table lists the three cases with the lowest context-recall proxy; it reports measured values without assigning an unverified cause.

| Rank | Question | Configuration | Answer relevance proxy | Context recall proxy | Context precision proxy |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Người tiêu dùng nên làm gì để giảm rủi ro khi mua sắm trực tuyến? | Dense-only | 0.615 | 0.514 | 1.000 |
| 2 | Tổ chức, cá nhân bán lẻ phải thực hiện những trách nhiệm gì về an toàn và chất lượng hàng hóa? | Hybrid + RRF | 0.625 | 0.514 | 0.200 |
| 3 | Người tiêu dùng nên làm gì để giảm rủi ro khi mua sắm trực tuyến? | Hybrid + RRF | 0.615 | 0.543 | 1.000 |

## Limitations and interpretation

- The four measures are deterministic lexical/source proxies, not semantic judgments from Ragas or a human evaluator.
- Answers are extracted directly from retrieved chunks; the faithfulness proxy is therefore expected to be high and does not independently establish semantic correctness.
- Context precision is measured by expected source filename, not by expert relevance labels for each chunk.
- Threshold separation is measured on 16 in-domain and 8 fixed out-of-domain queries; this sample does not establish performance on unseen queries.
- Latency is one warmed local run over the golden set, not a repeated benchmark or production estimate.
