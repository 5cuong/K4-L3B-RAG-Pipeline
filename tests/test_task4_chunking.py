import pytest

from src.task4_chunking_indexing import (
    DOCUMENT_TYPE_CHUNKING,
    chunk_documents_by_type,
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
