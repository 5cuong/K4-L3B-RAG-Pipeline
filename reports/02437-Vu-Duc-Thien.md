# Individual contribution report

## Thông tin

- Họ và tên: Vũ Đức Thiện
- Mã học viên: 2A202602437
- Nhóm: TDCH
- Repository/branch: `K4-L3B-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 9 — Retrieval pipeline | `retrieve()`: chạy dense và BM25 với `top_k * 2`, fuse bằng RRF đúng một lần; quyết định fallback PageIndex theo cosine top-1 gốc của dense; bắt lỗi provider fallback để vẫn trả kết quả hybrid. Merge code Task 9 vào `main` khi giải quyết conflict. | `src/task9_retrieval_pipeline.py`; merge `5b85cb6` (code vào repo qua commit gộp `7390cc8`) | Done |
| Task 10 — Generation có citation | Viết `call_llm()` gọi OpenAI/Gemini/Anthropic theo `LLM_PROVIDER`/`LLM_MODEL`, báo lỗi rõ khi thiếu API key. Viết `generate_with_citation()` trả safe refusal khi query rỗng, lỗi retrieval, không có chunk, LLM lỗi hoặc trả rỗng. Thêm chunk ID vào `format_context()` và vào prompt bắt buộc citation. | `src/task10_generation.py`; commit `433bd72` | Done |
| Streamlit UI | Nối `app.py` với `generate_with_citation()`; thêm `render_sources()` hiển thị retrieval method, title, score, chunk ID và nội dung; lưu `result` vào session state để lịch sử chat vẫn hiện nguồn. | `app.py`; commit `433bd72` | Done |

Thành viên khác bổ sung sau, không kê khai là phần của tôi: nhãn `[S1]` và `_expand_citation_aliases()`, hàm kiểm tra `citations_map_to_sources()`, provider Groq, `SCORE_THRESHOLD` đọc từ `.env` (`af6d101`, `ea85e12`).

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Task 9 so ngưỡng fallback với cosine similarity top-1 của dense search, không với RRF score.  
   **Lý do/evidence:** RRF score = Σ 1/(60 + rank), tối đa khoảng 0,033 với hai danh sách, nên không cùng thang với cosine. Nếu so với ngưỡng 0,5 thì mọi query đều rơi vào fallback. Test `test_retrieve_uses_dense_score_for_fallback` (dense 0,2 → PageIndex) và `test_retrieve_fuses_once_when_dense_is_confident` (dense 0,9 → RRF được gọi đúng một lần, không gọi fallback).  
   **Trade-off:** Quyết định chỉ dựa vào một điểm dense top-1. Query khớp mạnh theo từ khóa (BM25) nhưng dense yếu vẫn bị chuyển sang PageIndex. Khi fallback có kết quả, kết quả hybrid bị bỏ hẳn thay vì trộn với kết quả fallback.

2. **Quyết định:** `generate_with_citation()` không bao giờ raise; mọi lỗi đều trả về `GenerationResult` hợp lệ với `SAFE_REFUSAL`. Context gửi LLM được reorder để giảm lost-in-the-middle, nhưng `sources` giữ thứ tự score giảm dần; mỗi đoạn context mang chunk ID để map ngược citation.  
   **Lý do/evidence:** UI và `validate_generation_result` luôn cần đủ `answer`/`sources`/`retrieval_source`. Provider chưa có key (`.env` trên máy tôi chưa có key LLM) không được làm crash app.  
   **Trade-off:** Người dùng thấy cùng một câu từ chối cho cả trường hợp "thiếu evidence" lẫn "provider lỗi", nên khó debug. Khi LLM lỗi, `sources=[]` nên UI không hiện các chunk đã retrieve.

## Kiểm thử và kết quả

- Test/query đã dùng: `python -m pytest tests/test_contracts.py -v -k "retrieve or reorder or generation or call_llm or signatures"` → **8 passed**, gồm 3 test fallback/RRF của Task 9, reorder và `format_context`, alias citation, dispatch Groq, chữ ký hàm, safe refusal. Toàn bộ `pytest -q` → **28 passed** (2026-09-25).
- Script mock `retrieve`/`call_llm` kiểm 7 nhánh của `generate_with_citation()`, cả 7 kết quả đều qua `validate_generation_result`:
  - Query rỗng, lỗi retrieval, không có chunk, lỗi LLM → refusal, `retrieval_source="none"`.
  - Answer không có citation, hoặc có citation không tồn tại `[S9]` → refusal, vẫn giữ `sources`.
  - Answer có `[S1]` → được thay bằng `[legal/a.md::chunk-0]`.
- Trước/sau: trước `433bd72`, `call_llm()` và `generate_with_citation()` raise `NotImplementedError` và UI chỉ in placeholder `TODO`. Sau commit, UI trả lời kèm panel nguồn. Chưa có số liệu evaluation cho generation với LLM thật.
- Lỗi đã phát hiện và cách xử lý: bản đầu của tôi chỉ yêu cầu citation `[chunk_id]` qua prompt, không hậu kiểm, nên LLM có thể rút gọn hoặc tự tạo ID. Nhóm đã bổ sung nhãn ngắn `[S1]` được expand về chunk ID và từ chối trả lời khi citation không khớp nguồn (`af6d101`; test `test_generation_expands_short_citations_to_stable_chunk_ids`).

## Điều còn hạn chế

- Hạn chế cụ thể:
  - Chưa chạy end-to-end với LLM và ChromaDB thật trên máy tôi (`.env` chưa có key LLM, chưa build `chroma_db`).
  - `tests/` chưa có test tự động cho các nhánh của `generate_with_citation()`.
  - Evaluation A/B của nhóm tự ghép dense + RRF chứ không gọi `retrieve()`, nên nhánh fallback PageIndex của Task 9 chưa được đo.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: chuyển 7 nhánh mock ở trên thành test pytest, rồi chạy `generate_with_citation()` với LLM thật trên 16 câu của `golden_dataset.json` để đo tỷ lệ citation hợp lệ và tỷ lệ refusal.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Vũ Đức Thiện
