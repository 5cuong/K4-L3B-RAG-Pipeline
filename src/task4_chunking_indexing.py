"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

from functools import lru_cache
import os
import re
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Dữ liệu có hai dạng chính: văn bản pháp lý nhiều tầng mục/điều và bài báo
# có heading, đoạn văn cùng phần HTML navigation. Giữ cấu hình riêng giúp
# chunk giữ được ngữ nghĩa và có thể trình bày rõ trong báo cáo.
DOCUMENT_TYPE_CHUNKING = {
    "legal": {"chunk_size": 1200, "chunk_overlap": 150},
    "news": {"chunk_size": 700, "chunk_overlap": 80},
}

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

COLLECTION_NAME = "rag_documents"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with the provider configured in ``.env``."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL).strip()

    if provider == "sentence_transformers":
        model = _get_sentence_transformer(model_name)
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=model_name, input=texts)
        data = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in data]

    if provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for EMBEDDING_PROVIDER=gemini")
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(model=model_name, contents=texts)
        return [
            [float(value) for value in embedding.values]
            for embedding in response.embeddings
        ]

    raise ValueError(
        "Unsupported EMBEDDING_PROVIDER: "
        f"{provider!r}. Use sentence_transformers, openai or gemini."
    )


@lru_cache(maxsize=2)
def _get_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    model_path = model_name
    if not Path(model_name).is_dir():
        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_name, local_files_only=True)
        except Exception:
            # Fall back to the model identifier so a normal online setup can
            # download it when no local snapshot is available.
            model_path = model_name

    return SentenceTransformer(model_path)


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    paths = sorted(
        (path for path in STANDARDIZED_DIR.rglob("*.md") if path.is_file()),
        key=lambda path: path.relative_to(STANDARDIZED_DIR).as_posix().lower(),
    )
    for path in paths:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = path.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if "legal" in relative_path.parts else "news"
        title_match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
        source_match = re.search(r"(?mi)^\*\*Source:\*\*\s*(\S+)\s*$", content)
        url = source_match.group(1) if source_match else None
        if url:
            from .task3_convert_markdown import canonical_source_url

            url = canonical_source_url(url)

        # The title and provenance are represented in metadata, not repeated
        # as retrieval text in every first chunk.
        content = re.sub(
            r"(?mi)^\*\*(?:Source|Crawled|Source document):\*\*.*(?:\n|$)",
            "",
            content,
        )
        content = re.sub(r"(?m)^---\s*\n", "", content, count=1).strip()
        if title_match and title_match.start() == 0:
            content = content.replace(title_match.group(0), "", 1).strip()
        documents.append(
            {
                "id": relative_path.as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": title_match.group(1).strip() if title_match else path.stem,
                    "doc_type": doc_type,
                    "url": url,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    if not documents:
        return []
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_SIZE must be positive and greater than CHUNK_OVERLAP")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        content = str(document.get("content", "")).strip()
        if not content:
            continue
        for index, text in enumerate(splitter.split_text(content)):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {
                        **document["metadata"],
                        "chunk_index": index,
                    },
                }
            )
    return chunks


def chunk_documents_by_type(documents: list[dict]) -> list[dict]:
    """Chunk theo cấu trúc và loại tài liệu, độc lập với ``chunk_documents``.

    Legal documents ưu tiên ranh giới Chương/Mục/Điều/Khoản và context lớn.
    News documents ưu tiên heading/đoạn văn và context nhỏ hơn để hạn chế trộn
    nội dung bài báo với phần điều hướng được crawl từ HTML.
    """
    boundary_patterns = {
        "legal": [
            r"\n(?=\*\*(?:Chương|CHƯƠNG|Mục|MỤC|Điều|ĐIỀU|Khoản|KHOẢN)\b)",
            r"\n(?=\d+\.\s)",
            r"\n\s*\n",
            r"\n",
            r"(?<=[.!?;:])\s+",
            r"\s+",
        ],
        "news": [
            r"\n(?=#{1,6}\s)",
            r"\n\s*\n",
            r"\n",
            r"(?<=[.!?])\s+",
            r"\s+",
        ],
    }

    chunks = []
    for document in documents:
        doc_type = str(document.get("metadata", {}).get("doc_type", "news")).lower()
        if doc_type not in DOCUMENT_TYPE_CHUNKING:
            raise ValueError(f"Unsupported doc_type: {doc_type!r}")

        config = DOCUMENT_TYPE_CHUNKING[doc_type]
        texts = _split_text_at_boundaries(
            str(document.get("content", "")),
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            boundary_patterns=boundary_patterns[doc_type],
        )
        for index, text in enumerate(texts):
            if not text.strip():
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {
                        **document["metadata"],
                        "chunk_index": index,
                    },
                }
            )
    return chunks


def _split_text_at_boundaries(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    boundary_patterns: list[str],
) -> list[str]:
    """Split tại ranh giới ưu tiên, fallback về giới hạn ký tự cứng."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and greater than chunk_overlap")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end

        if hard_end < len(text):
            window = text[start:hard_end]
            minimum_boundary = max(chunk_size // 2, chunk_overlap + 1)
            for pattern in boundary_patterns:
                candidates = [
                    match.start()
                    for match in re.finditer(pattern, window)
                    if match.start() >= minimum_boundary
                ]
                if candidates:
                    end = start + candidates[-1]
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(0, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []

    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding provider returned a different number of vectors")

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    collection = get_collection()
    metadata = []
    for chunk in chunks:
        item = dict(chunk["metadata"])
        # Chroma metadata values cannot be None. Keep the field as an empty
        # string so query results can be normalized back to the contract.
        if item.get("url") is None:
            item["url"] = ""
        metadata.append(item)

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=metadata,
    )


def prune_stale_chunks(active_chunk_ids: set[str]) -> int:
    """Remove IDs left behind when a full rebuild produces fewer chunks."""
    collection = get_collection()
    existing_ids = collection.get(include=["metadatas"]).get("ids", [])
    stale_ids = [item_id for item_id in existing_ids if item_id not in active_chunk_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)
    return len(stale_ids)


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No standardized Markdown documents found in {STANDARDIZED_DIR}")
    chunks = chunk_documents_by_type(documents)
    if not chunks:
        raise RuntimeError("The current corpus produced no chunks; existing index was kept")
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    removed = prune_stale_chunks({chunk["id"] for chunk in embedded_chunks})
    print(f"Indexed {len(embedded_chunks)} chunks; removed {removed} stale chunks")


if __name__ == "__main__":
    run_pipeline()
