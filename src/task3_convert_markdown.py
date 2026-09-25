"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Official Government legal-document records used as the source of the local
# DOCX corpus. Keeping title and URL beside the conversion code makes a fresh
# conversion retain source provenance in the standardized Markdown.
LEGAL_DOCUMENT_SOURCES = {
    "19_2023_QH15_m_500102": (
        "Luật số 19/2023/QH15 — Luật Bảo vệ quyền lợi người tiêu dùng",
        "https://vanban.chinhphu.vn/?docid=208363&pageid=27160",
    ),
    "55_2024_ND-CP_m_610488": (
        "Nghị định số 55/2024/NĐ-CP — Quy định chi tiết một số điều của Luật Bảo vệ quyền lợi người tiêu dùng",
        "https://vanban.chinhphu.vn/?docid=210254&pageid=27160",
    ),
    "122_2025_QH15_m_662035": (
        "Luật số 122/2025/QH15 — Luật Thương mại điện tử",
        "https://vanban.chinhphu.vn/?classid=1&docid=216503&pageid=27160&typegroupid=3",
    ),
    "248_2026_ND-CP_m_713280": (
        "Nghị định số 248/2026/NĐ-CP — Quy định chi tiết một số điều của Luật Thương mại điện tử",
        "https://vanban.chinhphu.vn/?docid=218747&pageid=27160&typegroupid=4",
    ),
}


def canonical_source_url(url: str) -> str:
    """Drop analytics parameters while preserving meaningful query values."""
    parts = urlsplit(url.strip())
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parts.query)
            if not key.lower().startswith("utm_")
        ]
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def convert_legal_docs() -> None:
    """Convert legal PDF/DOC/DOCX files into standardized Markdown."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = MarkItDown()
    supported_extensions = {".pdf", ".doc", ".docx"}

    for path in sorted(legal_dir.iterdir(), key=lambda item: item.name.lower()):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.suffix.lower() not in supported_extensions
        ):
            continue

        result = converter.convert(str(path))
        text_content = str(getattr(result, "text_content", "") or "").strip()
        if not text_content:
            continue

        title, source_url = LEGAL_DOCUMENT_SOURCES.get(
            path.stem,
            (path.stem.replace("_", " "), ""),
        )
        header = f"# {title}\n\n"
        if source_url:
            header += f"**Source:** {source_url}\n\n---\n\n"

        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(header + text_content + "\n", encoding="utf-8")


def convert_news_articles() -> None:
    """Convert crawled news JSON records into standardized Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    required_fields = ("title", "url", "date_crawled", "content_markdown")
    for path in sorted(news_dir.glob("*.json"), key=lambda item: item.name.lower()):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue

        values = {field: data.get(field) for field in required_fields}
        if not all(isinstance(value, str) and value.strip() for value in values.values()):
            continue
        values["url"] = canonical_source_url(values["url"])

        header = (
            f"# {values['title'].strip()}\n\n"
            f"**Source:** {values['url'].strip()}\n\n"
            f"**Crawled:** {values['date_crawled'].strip()}\n\n"
            "---\n\n"
        )
        content = values["content_markdown"].strip()
        if not content:
            continue

        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(header + content + "\n", encoding="utf-8")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
