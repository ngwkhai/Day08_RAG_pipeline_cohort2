"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _clean_markdown_outputs(output_dir: Path):
    """Remove generated markdown files so each run reflects current landing data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for md_file in output_dir.glob("*.md"):
        md_file.unlink()


def _unique_output_path(filepath: Path, output_dir: Path, used_stems: set[str]) -> Path:
    """Avoid overwriting when PDF/DOCX versions share the same source stem."""
    stem = filepath.stem
    if stem in used_stems:
        stem = f"{stem}-{filepath.suffix.lower().lstrip('.')}"

    base = stem
    counter = 2
    while stem in used_stems:
        stem = f"{base}-{counter}"
        counter += 1

    used_stems.add(stem)
    return output_dir / f"{stem}.md"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    _clean_markdown_outputs(output_dir)

    md = MarkItDown()
    used_stems: set[str] = set()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            # TODO: Convert và lưu file
            result = md.convert(str(filepath))
            output_path = _unique_output_path(filepath, output_dir, used_stems)
            output_path.write_text(result.text_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    _clean_markdown_outputs(output_dir)
    used_stems: set[str] = set()

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            # TODO: Đọc JSON, extract content_markdown, lưu thành .md
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = _unique_output_path(filepath, output_dir, used_stems)

            # Thêm metadata header
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
