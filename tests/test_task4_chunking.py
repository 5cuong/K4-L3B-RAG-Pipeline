import pytest

from src.task4_chunking_indexing import (
    DOCUMENT_TYPE_CHUNKING,
    chunk_documents_by_type,
    load_documents,
    prune_stale_chunks,
)


def make_document(doc_id: str, doc_type: str, content: str) -> dict:
    return {
        "id": doc_id,
        "content": content,
        "metadata": {
            "source": f"{doc_id}.md",
            "title": doc_id,
            "doc_type": doc_type,
            "url": None,
        },
    }


def test_chunk_documents_by_type_preserves_contract_and_uses_type_limits():
    documents = [
        make_document(
            "regulation",
            "legal",
            "# Quy định\n\nĐiều 1. " + "Nội dung pháp lý. " * 150,
        ),
        make_document(
            "article",
            "news",
            "# Tin tức\n\n" + "Thông tin trong bài báo. " * 100,
        ),
    ]

    chunks = chunk_documents_by_type(documents)

    assert chunks
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    for document in documents:
        document_chunks = [
            chunk for chunk in chunks if chunk["id"].startswith(f"{document['id']}::")
        ]
        limit = DOCUMENT_TYPE_CHUNKING[document["metadata"]["doc_type"]]["chunk_size"]
        assert document_chunks
        for index, chunk in enumerate(document_chunks):
            assert chunk["content"].strip()
            assert len(chunk["content"]) <= limit
            assert chunk["metadata"]["chunk_index"] == index
            assert chunk["metadata"]["source"] == document["metadata"]["source"]


def test_chunk_documents_by_type_rejects_unknown_document_type():
    document = make_document("unknown", "transcript", "Some content")

    with pytest.raises(ValueError, match="Unsupported doc_type"):
        chunk_documents_by_type([document])


def test_load_documents_keeps_clean_source_url_in_metadata(tmp_path, monkeypatch):
    import src.task4_chunking_indexing as task4

    news_dir = tmp_path / "news"
    news_dir.mkdir()
    (news_dir / "article.md").write_text(
        "# Article title\n\n"
        "**Source:** https://example.org/story?utm_source=tracker&id=4\n\n"
        "**Crawled:** 2026-09-25\n\n---\n\nArticle body text.",
        encoding="utf-8",
    )
    monkeypatch.setattr(task4, "STANDARDIZED_DIR", tmp_path)

    [document] = load_documents()

    assert document["metadata"]["title"] == "Article title"
    assert document["metadata"]["url"] == "https://example.org/story?id=4"
    assert "utm_source" not in document["content"]
    assert "Article title" not in document["content"]
    assert "Article body text." in document["content"]


def test_prune_stale_chunks_removes_ids_from_an_older_chunking_run(monkeypatch):
    import src.task4_chunking_indexing as task4

    class FakeCollection:
        def __init__(self):
            self.deleted = []

        def get(self, include):
            assert include == ["metadatas"]
            return {"ids": ["legal::chunk-0", "legal::chunk-1", "news::chunk-0"]}

        def delete(self, ids):
            self.deleted.extend(ids)

    collection = FakeCollection()
    monkeypatch.setattr(task4, "get_collection", lambda: collection)

    removed = prune_stale_chunks({"legal::chunk-0", "news::chunk-0"})

    assert removed == 1
    assert collection.deleted == ["legal::chunk-1"]
