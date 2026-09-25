# Individual contribution report

## Thông tin

- Họ và tên: Nguyễn Hoàng Cường
- Mã học viên/sinh viên: 02473
- Nhóm: TDCH
- Repository/branch: `K4-L3B-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 4 — Chunking, embedding và indexing | Xây dựng bước đọc tài liệu Markdown, chia chunk, tạo embedding và upsert vào ChromaDB. Điều chỉnh cách chia theo loại tài liệu: legal ưu tiên ranh giới cấu trúc và chunk lớn hơn; news ưu tiên heading/đoạn văn và chunk nhỏ hơn. | `src/task4_chunking_indexing.py` | Done |
| Task 5 — Semantic search | Tái sử dụng `embed_texts()` và collection cosine của Task 4 để embed query và tìm kiếm dense; đổi cosine distance thành score giảm dần. Chuẩn hóa metadata sau khi đọc từ ChromaDB, gồm khôi phục `url` rỗng về `None`, và loại ID trùng. | `src/task5_semantic_search.py` | Done |

## Quyết định kỹ thuật quan trọng

1. **Chia chunk theo loại tài liệu:** cấu hình legal là 1200 ký tự/150 overlap và news là 700/80; ưu tiên các ranh giới điều khoản pháp lý hoặc heading/đoạn báo. **Lý do:** hai loại tài liệu có cấu trúc khác nhau. **Trade-off:** chunk legal dài hơn có thể chứa nhiều ngữ cảnh nhưng tốn embedding hơn và kém chi tiết hơn khi truy xuất.
2. **Dùng chung embedding giữa Task 4 và Task 5:** Task 5 gọi `embed_texts()` của Task 4 và truy vấn đúng collection ChromaDB dùng cosine. **Lý do:** query và chunks phải ở cùng không gian vector. **Trade-off:** chất lượng xếp hạng phụ thuộc model/provider được cấu hình; cosine score chưa phải xác suất đúng.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest tests/test_contracts.py::test_semantic_search_uses_shared_embedding_and_contract -q` — **1 passed** với embedding và ChromaDB được mock.
- Kết quả trước/sau nếu có: chưa có số liệu evaluation trước/sau hoặc kết quả query với model/ChromaDB thật.
- Lỗi đã phát hiện và cách xử lý: Task 4 lưu `url=None` dưới dạng chuỗi rỗng vì ChromaDB không nhận metadata `None`; Task 5 đổi chuỗi rỗng về `None` khi tạo SearchResult. Task 5 cũng bỏ kết quả trùng ID.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: chưa đánh giá chất lượng truy xuất trên các query đại diện; cosine score chưa được hiệu chỉnh thành xác suất đúng.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: chạy evaluation trên query legal/news, rồi điều chỉnh kích thước chunk hoặc ngưỡng cosine theo kết quả.

## Xác nhận đóng góp

Tôi xác nhận báo cáo này phản ánh phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Nguyễn Hoàng Cường
