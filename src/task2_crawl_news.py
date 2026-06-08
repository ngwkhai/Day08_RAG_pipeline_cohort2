"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://baolaocai.vn/bao-dong-tinh-trang-nghe-si-dung-ma-tuy-va-nhung-he-luy-voi-xa-hoi-post900028.html",
    "https://nld.com.vn/showbiz-viet-nhung-nghe-si-gay-soc-vi-be-boi-ma-tuy-196250725113547841.htm",
    "https://cuoi.tuoitre.vn/loat-nghe-si-viet-tieu-tan-su-nghiep-vi-ma-tuy-20241114142620463.htm",
    "https://www.nguoiduatin.vn/loat-sao-viet-vuong-vong-lao-ly-vi-ma-tuy-hao-quang-khong-phai-la-chan-cho-chat-cam-204260515121550023.htm",
    "https://laodong.vn/van-hoa-giai-tri/miu-le-bi-dieu-tra-lien-quan-toi-ma-tuy-danh-tieng-kho-xay-nhung-de-mat-1700317.ldo",
]


def safe_filename(text: str) -> str:
    """
    Chuyển title thành tên file hợp lệ.
    """

    text = re.sub(r'[\\/*?:"<>|]', "_", text)
    text = text.strip()

    if len(text) > 120:
        text = text[:120]

    return text


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str,
            "content_markdown": str
        }
    """

    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:

        result = await crawler.arun(url=url)

        title = (
            getattr(result, "title", None)
            or result.metadata.get("title", "Unknown")
            if hasattr(result, "metadata")
            else "Unknown"
        )

        markdown = (
            getattr(result, "markdown", None)
            or getattr(result, "cleaned_markdown", "")
            or ""
        )

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": markdown,
        }


async def crawl_all():
    """Crawl toàn bộ ARTICLE_URLS."""

    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, start=1):

        try:

            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

            article = await crawl_article(url)

            title = article["title"]

            filename = (
                safe_filename(title)
                if title and title != "Unknown"
                else f"article_{i:02d}"
            )

            filepath = DATA_DIR / f"{filename}.json"

            filepath.write_text(
                json.dumps(
                    article,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            print(f"  ✓ Saved: {filepath}")

        except Exception as e:

            print(f"  ✗ Error: {e}")


if __name__ == "__main__":

    if not ARTICLE_URLS:

        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")

    else:

        asyncio.run(crawl_all())
