"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key → PAGEINDEX_API_KEY trong .env
    3. Upload PDF từ data/landing/legal/
    4. Query bằng submit_query + get_retrieval (vectorless retrieval API)
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from pageindex import PageIndexClient
from pageindex.client import PageIndexAPIError

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_LEGAL_DIR = _ROOT / "data" / "landing" / "legal"
DOC_IDS_FILE = _ROOT / "data" / "pageindex_doc_ids.json"

POLL_INTERVAL_SEC = 10
UPLOAD_TIMEOUT_SEC = 600
RETRIEVAL_TIMEOUT_SEC = 180

_client: PageIndexClient | None = None


def _get_client() -> PageIndexClient:
    global _client
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        raise ValueError(
            "PAGEINDEX_API_KEY chưa được cấu hình. "
            "Thêm key vào .env (lấy tại https://dash.pageindex.ai/api-keys)"
        )
    if _client is None:
        _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _client


def _load_doc_registry() -> list[dict]:
    if DOC_IDS_FILE.exists():
        return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    return []


def _save_doc_registry(registry: list[dict]):
    DOC_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_FILE.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _wait_for_document_ready(client: PageIndexClient, doc_id: str) -> dict:
    """Chờ document xử lý xong và sẵn sàng retrieval."""
    deadline = time.time() + UPLOAD_TIMEOUT_SEC
    while time.time() < deadline:
        info = client.get_document(doc_id)
        status = info.get("status", "")
        print(f"    status={status}", end="", flush=True)

        if status == "failed":
            raise RuntimeError(f"PageIndex xử lý thất bại: {doc_id}")

        if status == "completed" and client.is_retrieval_ready(doc_id):
            print(" → ready")
            return info

        print(" → chờ...", flush=True)
        time.sleep(POLL_INTERVAL_SEC)

    raise TimeoutError(f"Timeout chờ document {doc_id} sẵn sàng retrieval")


def upload_documents(force: bool = False) -> list[dict]:
    """
    Upload PDF pháp luật từ data/landing/legal/ lên PageIndex.

    PageIndex SDK chỉ nhận PDF. doc_id được lưu tại data/pageindex_doc_ids.json
    để tái sử dụng khi query, không upload lại mỗi lần chạy.
    """
    client = _get_client()
    registry = [] if force else _load_doc_registry()
    known_files = {item["file"] for item in registry}

    pdf_files = sorted(LANDING_LEGAL_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Không tìm thấy PDF trong {LANDING_LEGAL_DIR}")

    for pdf_path in pdf_files:
        if pdf_path.name in known_files and not force:
            print(f"  ↷ Đã upload trước đó: {pdf_path.name}")
            continue

        print(f"  ↑ Uploading: {pdf_path.name}")
        result = client.submit_document(str(pdf_path))
        doc_id = result["doc_id"]
        print(f"    doc_id={doc_id}")

        info = _wait_for_document_ready(client, doc_id)
        registry = [r for r in registry if r["file"] != pdf_path.name]
        registry.append({
            "file": pdf_path.name,
            "doc_id": doc_id,
            "name": info.get("name", pdf_path.name),
            "page_num": info.get("pageNum"),
        })
        _save_doc_registry(registry)
        print(f"  ✓ Ready: {pdf_path.name}")

    return registry


def get_doc_ids() -> list[str]:
    """Lấy danh sách doc_id đã upload."""
    registry = _load_doc_registry()
    if registry:
        return [item["doc_id"] for item in registry]

    # Chưa có registry local — thử lấy từ PageIndex cloud
    client = _get_client()
    remote = client.list_documents(limit=100).get("documents", [])
    completed = [
        d["id"] for d in remote
        if d.get("status") == "completed"
    ]
    if completed:
        return completed

    raise RuntimeError(
        "Chưa có document trên PageIndex. Chạy upload_documents() trước."
    )


def _poll_retrieval(client: PageIndexClient, retrieval_id: str) -> dict:
    deadline = time.time() + RETRIEVAL_TIMEOUT_SEC
    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        status = result.get("status", "")
        if status in ("completed", "success", "done"):
            return result
        if status in ("failed", "error"):
            raise RuntimeError(f"Retrieval thất bại: {result}")
        time.sleep(3)
    raise TimeoutError(f"Timeout retrieval {retrieval_id}")


def _parse_retrieval_result(result: dict, doc_id: str, file_name: str) -> list[dict]:
    """Chuẩn hóa response PageIndex thành format pipeline."""
    items: list[dict] = []

    # Format PageIndex retrieval API: retrieved_nodes
    nodes = result.get("retrieved_nodes")
    if isinstance(nodes, list):
        for rank, node in enumerate(nodes):
            score = max(0.1, 1.0 - rank * 0.05)
            title = node.get("title", "")
            node_id = node.get("id", "")

            for group in node.get("relevant_contents") or []:
                entries = group if isinstance(group, list) else [group]
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    content = entry.get("relevant_content") or entry.get("content") or ""
                    if not content.strip():
                        continue
                    items.append({
                        "content": content.strip(),
                        "score": score,
                        "metadata": {
                            "doc_id": doc_id,
                            "file": file_name,
                            "node_id": node_id,
                            "section_title": entry.get("section_title") or title,
                            "page": entry.get("physical_index"),
                        },
                        "source": "pageindex",
                    })
        return items

    # Format 1: result["result"] là list các chunk
    raw = result.get("result") or result.get("results") or result.get("data")
    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, str):
                content = item
                score = max(0.1, 1.0 - i * 0.05)
                meta = {"doc_id": doc_id, "file": file_name}
            elif isinstance(item, dict):
                content = (
                    item.get("content")
                    or item.get("text")
                    or item.get("chunk")
                    or ""
                )
                score = float(item.get("score", item.get("relevance", 1.0 - i * 0.05)))
                meta = {
                    "doc_id": doc_id,
                    "file": file_name,
                    "page": item.get("page") or item.get("page_index"),
                    "node_id": item.get("node_id"),
                }
            else:
                continue
            if content.strip():
                items.append({
                    "content": content.strip(),
                    "score": score,
                    "metadata": meta,
                    "source": "pageindex",
                })
        return items

    # Format 2: result["result"] là string (answer + context)
    if isinstance(raw, str) and raw.strip():
        return [{
            "content": raw.strip(),
            "score": 1.0,
            "metadata": {"doc_id": doc_id, "file": file_name},
            "source": "pageindex",
        }]

    # Format 3: answer field
    answer = result.get("answer") or result.get("response")
    if isinstance(answer, str) and answer.strip():
        return [{
            "content": answer.strip(),
            "score": 1.0,
            "metadata": {"doc_id": doc_id, "file": file_name},
            "source": "pageindex",
        }]

    return items


def _search_single_doc(
    client: PageIndexClient,
    doc_id: str,
    file_name: str,
    query: str,
    top_k: int,
) -> list[dict]:
    """Query một document qua retrieval API, fallback Chat API."""
    submitted = client.submit_query(doc_id=doc_id, query=query, thinking=False)
    retrieval_id = submitted["retrieval_id"]
    result = _poll_retrieval(client, retrieval_id)
    return _parse_retrieval_result(result, doc_id, file_name)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    client = _get_client()
    registry = _load_doc_registry()
    doc_map = {item["doc_id"]: item.get("file", "") for item in registry}

    if not doc_map:
        doc_ids = get_doc_ids()
        doc_map = {doc_id: "" for doc_id in doc_ids}

    all_results: list[dict] = []
    for doc_id, file_name in doc_map.items():
        if len(all_results) >= top_k:
            break
        if not client.is_retrieval_ready(doc_id):
            print(f"  ⚠ Document chưa sẵn sàng: {file_name or doc_id}")
            continue
        try:
            hits = _search_single_doc(client, doc_id, file_name, query, top_k)
            all_results.extend(hits)
        except PageIndexAPIError as exc:
            print(f"  ⚠ Query thất bại ({file_name or doc_id}): {exc}")
            if "Insufficient credits" in str(exc):
                break

    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Loại trùng nội dung
    seen: set[str] = set()
    unique: list[dict] = []
    for item in all_results:
        key = item["content"][:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
        sys.exit(1)

    print("=" * 50)
    print("Task 8: PageIndex Vectorless RAG")
    print("=" * 50)

    print("\n--- Upload documents ---")
    registry = upload_documents()
    print(f"\n✓ {len(registry)} documents trên PageIndex")

    print("\n--- Test query ---")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        src = r["metadata"].get("file", r["metadata"].get("doc_id", ""))
        print(f"[{r['score']:.3f}] ({src}) {r['content'][:120]}...")
