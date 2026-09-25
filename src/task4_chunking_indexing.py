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
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

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
        documents.append(
            {
                "id": relative_path.as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": path.stem,
                    "doc_type": doc_type,
                    "url": None,
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


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
