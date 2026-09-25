"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # TODO: Thêm ít nhất 5 public URL.
    "https://moit.gov.vn/tin-tuc/thi-truong-nuoc-ngoai/luu-y-khi-tien-hanh-cac-giao-dich-thuong-mai-dien-tu2.html?utm_source=chatgpt.com",
    "https://moit.gov.vn/tin-tuc/bao-chi-voi-nguoi-dan/bo-cong-thuong-day-manh-cong-tac-bao-ve-quyen-loi-nguoi-tieu-dung-trong-thuong-mai-dien-tu.html?utm_source=chatgpt.com",
    "https://moit.gov.vn/tin-tuc/bao-chi-voi-nguoi-dan/bo-cong-thuong-canh-bao-nguoi-tieu-dung-ve-rui-ro-khi-mua-sam-tren-cac-nen-tang-tmdt-xuyen-bien-gioi-chua-dang-ky.html?utm_source=chatgpt.com",
    "https://moit.gov.vn/tin-tuc/bao-chi-voi-nguoi-dan/mot-so-trach-nhiem-quan-trong-ve-bao-ve-quyen-loi-nguoi-tieu-dung-cua-cac-to-chuc-ca-nhan-phan-phoi-ban-le-hang-tieu-dun.html?utm_source=chatgpt.com",
    "https://moit.gov.vn/tin-tuc/bo-cong-thuong-pho-bien-luat-thuong-mai-dien-tu-va-nghi-dinh-so-248-2026-nd-cp.html?utm_source=chatgpt.com"
]


async def crawl_article(url: str) -> dict:
    """Crawl one public article and return the landing-page record."""
    # Import lazily so that the rest of the project (and its tests) can still
    # import this module before Crawl4AI/Playwright has been installed.
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") or "Unknown"
    content_markdown = getattr(result, "markdown", "") or ""

    return {
        "url": url,
        "title": str(title).strip(),
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": str(content_markdown).strip(),
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
