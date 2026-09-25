# RAG evaluation results

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-25T07:40:19.224146+00:00 |
| Framework and version | Offline evaluator in `run_ab_evaluation.py` |
| Evaluator model | None; deterministic token-overlap proxy |
| Generator model | Deterministic extractive baseline |
| Embedding model | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Corpus version/commit | 5f6049f |
| Golden dataset size | 16 |
| Indexed chunk count | 915 |
| `top_k` | 5 |
| Fallback threshold and calibration | configured=0.5731; recommended=0.5731; balanced accuracy=1.000 on 16 in-domain and 8 out-of-domain queries |

## Configurations

- **Config A — dense-only:** semantic search over ChromaDB.
- **Config B — hybrid + RRF:** dense search + BM25, fused once with RRF.

Both configurations use the same golden cases, top_k, document-type-aware chunks, embedding model, extractive generator and evaluator. Metrics are deterministic offline proxies, not Ragas or LLM-graded scores.

Task 4 uses legal chunks (1200 characters, 150 overlap) and news chunks (700 characters, 80 overlap). The active index contains 915 chunks; the prior recursive 500/50 strategy would produce 1649 chunks for the same Markdown corpus.

Fallback calibration used dense top-1 cosine similarity. In-domain score range: [0.6001, 0.8219]; out-of-domain score range: [0.253, 0.5462]. Review false positives/negatives below before changing `.env`.

- False-negative in-domain queries: 0.
- False-positive out-of-domain queries: 0.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 1.000 | 1.000 | 0.000 |
| Answer relevance | 0.746 | 0.775 | 0.030 |
| Context recall | 0.812 | 0.862 | 0.050 |
| Context precision | 1.000 | 1.000 | 0.000 |
| **Average** | **0.889** | **0.909** | **0.020** |

## A/B comparison

- **Cấu hình tốt hơn theo proxy trung bình:** Config B — hybrid + RRF.
- **Evidence:** hybrid recall=0.862, precision=1.000; dense recall=0.812, precision=1.000.
- **Latency:** dense-only 0.80s; hybrid+RRF 2.67s trên 16 query. Hybrid có thêm chi phí BM25 và fusion nhưng không gọi LLM/API ngoài.
- **Caveat:** proxy này dùng để kiểm tra retrieval và regression; kết quả Ragas cần chạy lại khi cài evaluator model.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Người tiêu dùng nên làm gì để giảm rủi ro khi mua sắm trực tuyến? | dense-only | 1.000 | 0.615 | 0.514 | 1.000 | data | Question evidence is distributed across long legal/news documents. |
| 2 | Nghị định 248/2026/NĐ-CP áp dụng cho những nhóm chủ thể nào? | dense-only | 1.000 | 0.545 | 0.643 | 1.000 | data | Question evidence is distributed across long legal/news documents. |
| 3 | Tổ chức, cá nhân bán lẻ phải thực hiện những trách nhiệm gì về an toàn và chất lượng hàng hóa? | hybrid-rrf | 1.000 | 0.625 | 0.514 | 1.000 | data | Question evidence is distributed across long legal/news documents. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| ---: | --- | --- | --- | --- |
| 1 | Thử lọc boilerplate của news trước khi chunking | Hiện có 915 document-type chunks; soát worst performers ở trên để xác định phần nav gây nhiễu | Tăng mật độ evidence trong top-k và giảm chi phí embedding | Đo lại 4 metric trên cùng golden dataset |
| 2 | Bổ sung câu hỏi paraphrase và câu hỏi cần phân biệt nguồn | Bộ hiện tại có 16 câu; xem bảng worst performers để chọn khoảng trống | Đo được khả năng dense/BM25 rõ hơn | Thêm case vào golden dataset rồi chạy lại A/B |
| 3 | Chạy Ragas với cùng generator/evaluator khi môi trường đủ dependency | Proxy hiện tại không thay thế đánh giá LLM | Có faithfulness/relevance chuẩn hơn | Cài `ragas`, cấu hình evaluator và ghi lại model/version |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | --- | --- |
| Document-type-aware chunking | Recursive 500/50: 1649 chunks | Retrieval delta chưa đo riêng trong lần chạy này | -734 chunks trước khi embed | Giữ chiến lược theo loại tài liệu; A/B ở trên so sánh retriever trên cùng index |
