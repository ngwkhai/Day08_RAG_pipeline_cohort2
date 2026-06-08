"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    crawl4ai-setup   # cài Playwright browser (chạy lần đầu)
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Bài báo về nghệ sĩ Việt Nam liên quan ma tuý (VnExpress, Tuổi Trẻ)
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/su-nghiep-long-nhat-truoc-khi-bi-bat-vi-lien-quan-ma-tuy-5076081.html",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html",
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    "https://tuoitre.vn/khoi-to-3-bi-can-trong-vu-ca-si-miu-le-su-dung-ma-tuy-o-cat-ba-20260514230349573.htm",
    "https://tuoitre.vn/vu-miu-le-long-nhat-son-ngoc-minh-nghe-si-phai-giu-hinh-anh-chin-chu-tren-san-khau-lan-ngoai-doi-2026052112085492.htm",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str, max_len: int = 60) -> str:
    """Chuyển tiêu đề thành tên file an toàn."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        slug = "article"
    return slug[:max_len]


def filename_from_url(url: str, index: int) -> str:
    """Tạo tên file từ path URL hoặc index."""
    path = urlparse(url).path.strip("/")
    if path:
        stem = path.split("/")[-1].replace(".html", "").replace(".htm", "")
        stem = re.sub(r"[^\w-]", "-", stem)
        stem = re.sub(r"-+", "-", stem).strip("-")
        if stem:
            return f"{index:02d}-{stem}.json"
    return f"article_{index:02d}.json"


def extract_title(result, url: str) -> str:
    """Lấy tiêu đề từ metadata Crawl4AI hoặc markdown."""
    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") or metadata.get("og:title") or metadata.get("page_title")
    if title and title.strip() and title.strip().lower() not in {"unknown", "untitled"}:
        return title.strip()

    markdown = getattr(result, "markdown", "") or ""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return url


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)

        if not result.success:
            raise RuntimeError(
                f"Crawl thất bại: {url} — {getattr(result, 'error_message', 'unknown error')}"
            )

        content = (result.markdown or "").strip()
        if len(content) < 200:
            content = (getattr(result, "cleaned_html", None) or result.html or "").strip()

        return {
            "url": url,
            "title": extract_title(result, url),
            "date_crawled": datetime.now().isoformat(timespec="seconds"),
            "content_markdown": content,
        }


async def crawl_all(urls: list[str] | None = None):
    """Crawl toàn bộ bài báo trong ARTICLE_URLS (hoặc danh sách truyền vào)."""
    setup_directory()
    targets = urls if urls is not None else ARTICLE_URLS

    if not targets:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        return

    saved = 0
    for i, url in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ✗ Lỗi: {exc}")
            continue

        filename = filename_from_url(url, i)
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {filepath.name} — {article['title'][:70]}")
        saved += 1

    print(f"\nHoàn tất: {saved}/{len(targets)} bài đã lưu vào {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
