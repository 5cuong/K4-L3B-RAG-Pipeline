# Day 8 — RAG Pipeline: Bảo vệ người tiêu dùng và thương mại điện tử

## Mục tiêu và phạm vi

Repository triển khai chatbot RAG tiếng Việt để tra cứu quy định về bảo vệ người tiêu dùng và thương mại điện tử. Bộ dữ liệu hiện có gồm 4 văn bản pháp lý và 5 bài viết từ Bộ Công Thương; dữ liệu nguồn, URL, nội dung đã chuẩn hóa và bộ câu hỏi nằm trong repository.

## Dữ liệu

Văn bản pháp lý trong `data/landing/legal/`:

- Luật số 19/2023/QH15 về Bảo vệ quyền lợi người tiêu dùng.
- Nghị định số 55/2024/NĐ-CP quy định chi tiết một số điều của Luật Bảo vệ quyền lợi người tiêu dùng.
- Luật số 122/2025/QH15 về Thương mại điện tử.
- Nghị định số 248/2026/NĐ-CP quy định chi tiết một số điều của Luật Thương mại điện tử.

Năm bài viết crawl được lưu thành JSON tại `data/landing/news/`. URL nguồn được giữ trong dữ liệu và Markdown chuẩn hóa tại `data/standardized/news/`. Bốn văn bản pháp lý chuẩn hóa nằm tại `data/standardized/legal/`.

## Kiến trúc

`task3` chuẩn hóa tài liệu và loại bỏ phần điều hướng, chân trang khỏi nội dung bài báo. `task4` chia chunk theo loại tài liệu (legal: 1200 ký tự, overlap 150; news: 700 ký tự, overlap 80), tạo embedding đa ngôn ngữ bằng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều), rồi index trong ChromaDB với cosine distance. ChromaDB là dữ liệu sinh tại máy, không được commit.

Retrieval kết hợp dense search và BM25, sau đó hợp nhất một lần bằng Reciprocal Rank Fusion (RRF). Nếu cosine similarity dense top-1 thấp hơn `SCORE_THRESHOLD`, pipeline thử PageIndex fallback. Generation gọi provider được cấu hình qua `.env` (Groq dùng OpenAI-compatible API), gắn citation theo ID nguồn đã truy xuất và từ chối trả lời khi citation không ánh xạ được về nguồn. Giao diện Streamlit hiển thị câu trả lời và nguồn.

## Cài đặt và chạy lại

Yêu cầu Python 3.10–3.13. Các tài liệu nguồn đã có sẵn; Task 1 kiểm tra corpus hiện có. Chỉ chạy Task 2 khi cần crawl lại các bài viết.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Điền `GROQ_API_KEY` trong `.env` để dùng Groq; đặt `LLM_PROVIDER=groq` và `LLM_MODEL=openai/gpt-oss-20b`. Embedding mặc định chạy local và cần tải model ở lần chạy đầu. Không commit `.env` hoặc API key.

```bash
python -m src.task1_collect_legal_docs
python -m src.task3_convert_markdown
python -m src.task4_chunking_indexing
python -m src.calibrate_fallback_threshold
```

Lệnh calibration in ngưỡng đề xuất từ 16 câu hỏi in-domain và 8 câu hỏi out-of-domain; nó không tự ghi đè `.env`. Đặt `SCORE_THRESHOLD` theo kết quả calibration trước khi chạy chatbot hoặc evaluation. Nếu cần thu thập lại bài báo, cài Chromium bằng `python -m playwright install chromium`, đặt cấu hình crawler trong `.env`, rồi chạy `python -m src.task2_crawl_news` trước Task 3.

```bash
python -m group_project.evaluation.run_ab_evaluation
streamlit run app.py
pytest -q
```

## Đánh giá

`group_project/evaluation/golden_dataset.json` chứa 16 câu hỏi. Báo cáo `group_project/evaluation/RESULT.md` so sánh dense-only với hybrid + RRF trên cùng corpus, embedding, `top_k` và bộ câu hỏi.

Bốn chỉ số trong báo cáo là proxy offline dựa trên token overlap và tên file nguồn: faithfulness, answer relevance, context recall và context precision. Bộ sinh dùng chung cho A/B là extractive baseline xác định, không gọi LLM; do đó faithfulness proxy không chứng minh tính đúng ngữ nghĩa. Context precision đo tỷ lệ chunk đến từ file nguồn kỳ vọng, không phải đánh giá relevance thủ công. Báo cáo nêu rõ các giới hạn này và cách diễn giải kết quả.

## Tài liệu và deliverables

- [Hướng dẫn từng bước](docs/STEP_BY_STEP.md)
- [Module contracts](docs/MODULE_CONTRACTS.md)
- [Grading rubric](docs/GRADING_RUBRIC.md)
- [Báo cáo A/B](group_project/evaluation/RESULT.md)
- [Template báo cáo cá nhân](group_project/ịndividual/INDIVIDUAL_REPORT.md)
