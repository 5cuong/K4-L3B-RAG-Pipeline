"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import numpy as np


CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25Okapi(tokenized)


def _get_corpus() -> list[dict]:
    if CORPUS:
        return CORPUS

    from .task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    corpus = _get_corpus()
    if not corpus:
        return []

    query_tokens = query.lower().split()
    bm25 = build_bm25_index(corpus)
    scores = np.asarray(bm25.get_scores(query.lower().split()), dtype=float)
    indices = np.argsort(-scores, kind="stable")[:top_k]

    results = []
    seen_ids = set()
    for index in indices:
        score = float(scores[index])
        item = corpus[int(index)]
        document_tokens = set(item["content"].lower().split())
        # On very small corpora BM25 can assign a zero score to a matching
        # term because its IDF is zero. Keep actual lexical matches, but do
        # not return unrelated zero-score documents.
        if score <= 0 and not any(token in document_tokens for token in query_tokens):
            continue
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        metadata = dict(item["metadata"])
        metadata.setdefault("url", None)
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "metadata": metadata,
                "retrieval_method": "bm25",
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
