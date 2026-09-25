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


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


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

        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(text_content + "\n", encoding="utf-8")


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
