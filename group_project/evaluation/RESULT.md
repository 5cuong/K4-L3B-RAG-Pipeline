# RAG evaluation results

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-25T04:36:10.040197+00:00 |
| Framework and version | Offline evaluator in `run_ab_evaluation.py` |
| Evaluator model | None; deterministic token-overlap proxy |
| Generator model | Deterministic extractive baseline |
| Embedding model | BAAI/bge-m3 |
| Corpus version/commit | 7390cc8 |
| Golden dataset size | 16 |
| Indexed chunk count | 1206 |
| `top_k` | 5 |
| Fallback threshold and calibration | 0.5; calibrated earlier with in-domain ≈0.73 and out-of-domain ≈0.41 |

## Configurations

- **Config A — dense-only:** semantic search over ChromaDB.
- **Config B — hybrid + RRF:** dense search + BM25, fused once with RRF.

Both configurations use the same 16 cases, top_k, embedding model, extractive generator and evaluator. The metrics are offline proxies because `ragas` and a configured evaluator LLM are not installed in the current environment.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 1.000 | 1.000 | 0.000 |
| Answer relevance | 0.775 | 0.818 | 0.043 |
| Context recall | 0.856 | 0.800 | -0.055 |
| Context precision | 0.963 | 0.975 | 0.012 |
| **Average** | **0.898** | **0.898** | **-0.000** |

## A/B comparison

- **Cấu hình tốt hơn theo proxy trung bình:** Config A — dense-only.
- **Evidence:** hybrid recall=0.800, precision=0.975; dense recall=0.856, precision=0.963.
- **Latency:** dense-only 4.52s; hybrid+RRF 5.21s trên 16 query. Hybrid có thêm chi phí BM25 và fusion nhưng không gọi LLM/API ngoài.
- **Caveat:** proxy này dùng để kiểm tra retrieval và regression; kết quả Ragas cần chạy lại khi cài evaluator model.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Người tiêu dùng nên làm gì để giảm rủi ro khi mua sắm trực tuyến? | hybrid-rrf | 1.000 | 0.615 | 0.314 | 0.800 | retrieval | Expected evidence was not concentrated in the top-k retrieved chunks. |
| 2 | Theo Nghị định 55/2024/NĐ-CP, nền tảng số lớn được xác định theo tiêu chí nào? | hybrid-rrf | 1.000 | 0.615 | 0.621 | 1.000 | data | Question evidence is distributed across long legal/news documents. |
| 3 | Luật và Nghị định mới đặt ra yêu cầu gì đối với việc xác thực người bán trên nền tảng thương mại điện tử? | dense-only | 1.000 | 0.550 | 0.758 | 1.000 | data | Question evidence is distributed across long legal/news documents. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| ---: | --- | --- | --- | --- |
| 1 | Giữ lọc boilerplate của news trước khi chunking | News content đã giảm đáng kể và corpus hiện còn 1.206 chunks | Ít nhiễu, giảm chi phí embedding/retrieval | Chạy lại Task 3 và so sánh chunk count |
| 2 | Bổ sung các câu hỏi paraphrase và câu hỏi cần phân biệt nguồn | Worst cases thường có evidence nằm rải rác hoặc nhiều nguồn gần nghĩa | Đo được khả năng dense/BM25 rõ hơn | Thêm case vào golden dataset rồi chạy lại A/B |
| 3 | Chạy Ragas với cùng generator/evaluator khi môi trường đủ dependency | Proxy hiện tại không thay thế đánh giá LLM | Có faithfulness/relevance chuẩn hơn | Cài `ragas`, cấu hình evaluator và ghi lại model/version |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | --- | --- |
| News boilerplate filtering | Markdown raw từ crawler | Chunk count giảm từ khoảng 1.654 xuống 1.206 | Giảm số chunk news và chi phí xử lý | Nên giữ bước lọc trước Task 4 |
