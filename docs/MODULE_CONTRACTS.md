# Module contracts

Các nhóm có thể chọn thư viện khác nhau, nhưng input/output giữa các module phải thống nhất theo tài liệu này.

## Schema chung

```python
# Document hoặc chunk
{
    "id": str,
    "content": str,
    "metadata": {
        "source": str,
        "title": str,
        "doc_type": "legal" | "news",
        "url": str | None,
        "chunk_index": int       # chỉ bắt buộc sau khi chunk
    }
}

# SearchResult
{
    "id": str,
    "content": str,
    "score": float,
    "metadata": dict,
    "retrieval_method": "dense" | "bm25" | "hybrid" | "pageindex"
}

# GenerationResult
{
    "answer": str,
    "sources": list[SearchResult],
    "retrieval_source": "hybrid" | "pageindex" | "none"
}
```

## Interface bắt buộc

```python
# Task 4
load_documents() -> list[Document]
chunk_documents(documents) -> list[Document]
embed_texts(texts) -> list[list[float]]
embed_chunks(chunks) -> list[EmbeddedDocument]
get_collection()
index_to_vectorstore(chunks) -> None

# Task 5–8
semantic_search(query, top_k=10) -> list[SearchResult]
lexical_search(query, top_k=10) -> list[SearchResult]
rerank_rrf(ranked_lists, top_k=5, k=60) -> list[SearchResult]
pageindex_search(query, top_k=5) -> list[SearchResult]

# Task 9–10
retrieve(query, top_k=5, score_threshold=..., use_reranking=True) -> list[SearchResult]
reorder_for_llm(chunks) -> list[SearchResult]
format_context(chunks) -> str
generate_with_citation(query, top_k=5) -> GenerationResult
```

## Quy tắc bắt buộc

- `id` phải duy nhất và ổn định; metadata nguồn được giữ xuyên suốt pipeline.
- Chunk không rỗng, có `chunk_index`; chạy index lại không tạo dữ liệu trùng.
- Task 4 và Task 5 dùng chung embedding model, dimension và `embed_texts()`.
- Search result không trùng ID, không vượt `top_k` và được sort theo score giảm dần.
- RRF dùng công thức `sum(1 / (k + rank))`, rank bắt đầu từ 1 và chỉ fuse một lần.
- Fallback so sánh threshold với cosine score gốc của dense search, không dùng RRF score.
- PageIndex/provider lỗi không được làm UI crash; pipeline trả hybrid result hoặc safe refusal.
- Citation phải đối chiếu được với phần tử trong `sources`.
- Không hard-code hoặc commit API key.
- `tests/test_contracts.py` không được gọi network hay API thật.

Chạy kiểm tra:

```bash
pytest tests/test_contracts.py -q
```
