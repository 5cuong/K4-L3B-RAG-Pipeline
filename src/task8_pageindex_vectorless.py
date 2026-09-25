"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import requests


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROJECT_ROOT = Path(__file__).parent.parent
PAGEINDEX_PDF_DIR = PROJECT_ROOT / "pageindex_pdfs"
PAGEINDEX_BASE_URL = os.getenv("PAGEINDEX_BASE_URL", "https://api.pageindex.ai")
PAGEINDEX_CACHE = PROJECT_ROOT / "pageindex_doc_ids.json"
PAGEINDEX_TIMEOUT = float(os.getenv("PAGEINDEX_TIMEOUT", "30"))
PAGEINDEX_MAX_WAIT = float(os.getenv("PAGEINDEX_MAX_WAIT", "120"))


def _headers() -> dict[str, str]:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is required for PageIndex fallback")
    return {"api_key": PAGEINDEX_API_KEY}


def _request(method: str, path: str, **kwargs) -> dict:
    response = requests.request(
        method,
        f"{PAGEINDEX_BASE_URL.rstrip('/')}/{path.lstrip('/')}",
        headers=_headers(),
        timeout=PAGEINDEX_TIMEOUT,
        **kwargs,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("PageIndex returned a non-object response")
    return payload


def _load_cache() -> dict[str, dict]:
    if not PAGEINDEX_CACHE.exists():
        return {}
    try:
        data = json.loads(PAGEINDEX_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache: dict[str, dict]) -> None:
    PAGEINDEX_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pdf_sources() -> list[Path]:
    landing_dir = PROJECT_ROOT / "data" / "landing"
    source_pdfs = [
        path
        for path in landing_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    source_stems = {path.stem.casefold() for path in source_pdfs}
    generated_pdfs = [
        _markdown_to_pdf(path)
        for path in sorted(
            (STANDARDIZED_DIR / "legal").glob("*.md"),
            key=lambda item: item.name.lower(),
        )
        if path.is_file() and path.stem.casefold() not in source_stems
    ]
    return sorted(
        {*source_pdfs, *generated_pdfs},
        key=lambda path: path.relative_to(PROJECT_ROOT).as_posix().lower(),
    )


def _unicode_font_path() -> Path:
    """Find a system font that can render Vietnamese text in generated PDFs."""
    configured = os.getenv("PAGEINDEX_FONT_PATH", "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return configured_path
        raise RuntimeError(
            f"PAGEINDEX_FONT_PATH does not point to a file: {configured_path}"
        )

    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "A Unicode TrueType font is required to prepare DOCX/Markdown files "
        "for PageIndex. Install DejaVu Sans or set PAGEINDEX_FONT_PATH."
    )


def _markdown_to_pdf(markdown_path: Path) -> Path:
    """Create a cached PDF copy so PageIndex can index the DOCX corpus."""
    PAGEINDEX_PDF_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PAGEINDEX_PDF_DIR / f"{markdown_path.stem}.pdf"
    if (
        output_path.is_file()
        and output_path.stat().st_mtime_ns >= markdown_path.stat().st_mtime_ns
    ):
        return output_path

    try:
        from fpdf import FPDF
    except ImportError as error:
        raise RuntimeError(
            "fpdf2 is required to prepare the legal corpus for PageIndex; "
            "install project dependencies first."
        ) from error

    text = markdown_path.read_text(encoding="utf-8")
    text = "".join(
        character
        for character in text
        if character in "\n\r\t" or ord(character) >= 32
    )
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("CorpusUnicode", fname=str(_unicode_font_path()))
    pdf.set_font("CorpusUnicode", size=9)
    pdf.multi_cell(w=0, h=5, text=text)
    pdf.output(str(output_path))
    return output_path


def _source_key(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _display_source(path: Path) -> str:
    """Map generated PageIndex PDFs back to the checked-in source document."""
    if path.parent.resolve() == PAGEINDEX_PDF_DIR.resolve():
        legal_dir = PROJECT_ROOT / "data" / "landing" / "legal"
        for extension in (".pdf", ".docx", ".doc"):
            original = legal_dir / f"{path.stem}{extension}"
            if original.is_file():
                return _source_key(original)
        markdown = STANDARDIZED_DIR / "legal" / f"{path.stem}.md"
        if markdown.is_file():
            return _source_key(markdown)
    return _source_key(path)


def _cache_entry(path: Path, doc_id: str) -> dict:
    stat = path.stat()
    return {
        "doc_id": doc_id,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "source": _display_source(path),
    }


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    cache = _load_cache()
    changed = False

    for path in _pdf_sources():
        key = _source_key(path)
        stat = path.stat()
        cached = cache.get(key, {})
        if (
            cached.get("doc_id")
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
        ):
            continue

        with path.open("rb") as file_handle:
            response = requests.post(
                f"{PAGEINDEX_BASE_URL.rstrip('/')}/doc/",
                headers=_headers(),
                files={"file": (path.name, file_handle, "application/pdf")},
                data={"if_retrieval": "true"},
                timeout=PAGEINDEX_TIMEOUT,
            )
        response.raise_for_status()
        payload = response.json()
        doc_id = payload.get("doc_id") or payload.get("id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"PageIndex upload returned no document ID for {path.name}")

        cache[key] = _cache_entry(path, doc_id)
        changed = True

    if changed or not PAGEINDEX_CACHE.exists():
        _save_cache(cache)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if top_k <= 0 or not query.strip():
        return []

    upload_documents()
    cache = _load_cache()
    if not cache:
        return []

    retrieved = []
    for source, entry in cache.items():
        doc_id = entry.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            continue

        request = _request(
            "POST",
            "/retrieval/",
            json={"doc_id": doc_id, "query": query, "thinking": False},
        )
        retrieval_id = request.get("retrieval_id") or request.get("id")
        if not isinstance(retrieval_id, str) or not retrieval_id:
            continue

        result = _wait_for_retrieval(retrieval_id)
        display_source = entry.get("source") or source
        retrieved.extend(_parse_retrieval_result(result, display_source, doc_id))

    retrieved.sort(key=lambda item: item["score"], reverse=True)
    return retrieved[:top_k]


def _wait_for_retrieval(retrieval_id: str) -> dict:
    deadline = time.monotonic() + PAGEINDEX_MAX_WAIT
    while True:
        payload = _request("GET", f"/retrieval/{retrieval_id}/")
        status = str(payload.get("status", "")).lower()
        if status in {"completed", "complete", "success", "succeeded", "failed", "error"}:
            if status in {"failed", "error"}:
                raise RuntimeError("PageIndex retrieval failed")
            return payload
        if time.monotonic() >= deadline:
            raise TimeoutError("PageIndex retrieval timed out")
        time.sleep(1.0)


def _parse_retrieval_result(payload: dict, source: str, doc_id: str) -> list[dict]:
    candidates = payload.get("result")
    if candidates is None:
        candidates = payload.get("results")
    if candidates is None:
        candidates = payload.get("data")
    if isinstance(candidates, dict):
        candidates = (
            candidates.get("results")
            or candidates.get("nodes")
            or candidates.get("items")
            or [candidates]
        )
    if not isinstance(candidates, list):
        return []

    title = Path(source).stem
    doc_type = "news" if "news" in Path(source).parts else "legal"
    parsed = []
    for rank, item in enumerate(candidates, 1):
        if not isinstance(item, dict):
            continue
        content = item.get("text") or item.get("content") or item.get("markdown")
        if not isinstance(content, str) or not content.strip():
            continue

        page_index = item.get("page_index", item.get("page", 1))
        try:
            page_index = max(int(page_index), 1)
        except (TypeError, ValueError):
            page_index = 1

        raw_score = item.get("score", item.get("similarity", item.get("relevance_score")))
        try:
            score = float(raw_score) if raw_score is not None else 1.0 / rank
        except (TypeError, ValueError):
            score = 1.0 / rank

        node_id = item.get("node_id") or item.get("id") or f"page-{page_index}-{rank}"
        parsed.append(
            {
                "id": f"pageindex:{doc_id}:{node_id}",
                "content": content.strip(),
                "score": score,
                "metadata": {
                    "source": Path(source).name,
                    "title": title,
                    "doc_type": doc_type,
                    "url": None,
                    "chunk_index": page_index - 1,
                    "page_index": page_index,
                },
                "retrieval_method": "pageindex",
            }
        )
    return parsed


if __name__ == "__main__":
    upload_documents()
