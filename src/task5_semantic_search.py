"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    query_embeddings = embed_texts([query])
    if len(query_embeddings) != 1 or not query_embeddings[0]:
        raise ValueError("Embedding provider must return one non-empty query vector")
    query_vector = query_embeddings[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0] or []
    documents = response.get("documents", [[]])[0] or []
    metadatas = response.get("metadatas", [[]])[0] or []
    distances = response.get("distances", [[]])[0] or []

    results = []
    seen_ids = set()
    for item_id, content, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        if (
            not isinstance(item_id, str)
            or not item_id.strip()
            or item_id in seen_ids
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        seen_ids.add(item_id)
        normalized_metadata = dict(metadata or {})
        if normalized_metadata.get("url") == "":
            normalized_metadata["url"] = None
        else:
            normalized_metadata.setdefault("url", None)
        results.append(
            {
                "id": item_id,
                "content": content,
                "score": min(1.0, max(0.0, 1.0 - float(distance))),
                "metadata": normalized_metadata,
                "retrieval_method": "dense",
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
