# Individual contribution report

## Thông tin

- Họ và tên: Tống Trần Tiến Dũng
- Mã học viên: 2A202602791
- Nhóm: TDCH
- Repository/branch: `K4-L3B-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 2 — Crawl news | Khai báo 5 URL nguồn; dùng Crawl4AI để crawl từng bài; lưu `url`, `title`, `date_crawled` và `content_markdown` thành JSON. | `src/task2_crawl_news.py`, `data/landing/news/*.json` | Done |
| Task 3 — Chuẩn hóa Markdown | Chuyển legal và news sang Markdown, giữ cấu trúc `legal/` và `news/`; bổ sung metadata nguồn cho news; lọc phần điều hướng, quảng cáo và footer rác của trang web. | `src/task3_convert_markdown.py`, `data/standardized/legal/*.md`, `data/standardized/news/*.md` | Done |
| Task 4 — Chunking | Chia tài liệu bằng `RecursiveCharacterTextSplitter`; dùng ID ổn định dạng `source::chunk-index`, giữ metadata `source`, `title`, `doc_type`, `url` và `chunk_index`. | `src/task4_chunking_indexing.py`, `tests/test_task4_chunking.py` | Done |

## Quyết định kỹ thuật quan trọng

1. **Lọc nội dung thừa trước khi chunking.**
   **Lý do/evidence:** Nội dung crawl chứa breadcrumb, menu, banner, phần “Tin liên quan”, “Tags” và footer của website. Hàm `_clean_news_markdown()` giữ phần nội dung bài viết từ tiêu đề chính đến trước footer.
   **Trade-off:** Cần duy trì các mẫu nhận diện footer theo từng website, nhưng corpus ít nhiễu hơn và giảm số token/chunk phải embedding.

2. **Dùng recursive chunking với overlap.**
   **Lý do/evidence:** `chunk_documents()` dùng `RecursiveCharacterTextSplitter` với `CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 50` và các ranh giới ưu tiên là đoạn văn, dòng, câu rồi đến khoảng trắng. ID và `chunk_index` được tạo ổn định để chạy lại không đổi danh tính chunk.
   **Trade-off:** Chunk nhỏ giúp tìm đúng đoạn hơn nhưng làm tăng số lượng embedding; overlap giữ ngữ cảnh giữa hai chunk nhưng cũng làm tăng dữ liệu trùng lặp.

## Kiểm thử và kết quả

- Chạy `pytest -q`: **22 passed**.
- Acceptance test xác nhận có ít nhất 4 tài liệu legal, 5 bài news, Markdown chuẩn hóa và bộ golden dataset.
- `tests/test_task4_chunking.py` kiểm tra chunk không rỗng, ID duy nhất, giới hạn kích thước và metadata `chunk_index` liên tục.
- Chạy `python -m src.task3_convert_markdown` tạo Markdown trong đúng hai thư mục `data/standardized/legal/` và `data/standardized/news/`.
- Corpus hiện có 1.206 chunk theo cấu hình recursive baseline và metadata ChromaDB hợp lệ.

## Điều còn hạn chế

- Crawler hiện phụ thuộc vào Crawl4AI/Playwright và nguồn web bên ngoài; khi website thay đổi HTML, bộ lọc footer có thể cần cập nhật.
- Chunking hiện có cả `chunk_documents()` và biến thể `chunk_documents_by_type()`. Cần thống nhất một hàm dùng chung cho Task 4, BM25 và evaluation để dense và lexical retrieval chạy trên cùng corpus.
- Các chỉ số đánh giá hiện tại dùng proxy token-overlap khi môi trường chưa có Ragas và evaluator LLM; cần chạy lại evaluation bằng evaluator chính thức nếu được yêu cầu.

Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện là gom logic chuẩn hóa/chunking vào một pipeline dùng chung và thêm test hồi quy cho các mẫu HTML rác của từng nguồn news.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc mình đã thực hiện và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: _Bổ sung trước khi nộp_
