"""
RAG service layer — wrapper cho Task 9/10 với conversation memory.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from src.task10_generation import generate_with_citation


def build_follow_up_query(query: str, history: list[dict], max_turns: int = 3) -> str:
    """
    Ghép ngữ cảnh hội thoại gần nhất để hỗ trợ câu hỏi follow-up.

    Args:
        query: Câu hỏi hiện tại
        history: List of {'query', 'answer', 'sources', 'retrieval_source'}
        max_turns: Số lượt hội thoại gần nhất giữ lại
    """
    if not history:
        return query

    recent = history[-max_turns:]
    lines = ["=== Lịch sử hội thoại gần đây ==="]
    for i, turn in enumerate(recent, 1):
        lines.append(f"[Lượt {i}] Người dùng: {turn['query']}")
        answer_preview = (turn.get("answer") or "")[:400]
        lines.append(f"[Lượt {i}] Trợ lý: {answer_preview}")
    lines.append("=== Câu hỏi hiện tại ===")
    lines.append(query)
    lines.append(
        "\n(Lưu ý: nếu câu hỏi là follow-up như 'còn hình phạt thì sao?', "
        "hãy hiểu dựa trên lịch sử hội thoại phía trên.)"
    )
    return "\n".join(lines)


def check_pipeline_ready() -> tuple[bool, str]:
    """Kiểm tra ChromaDB và API key trước khi chat."""
    import os

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-xxx":
        return False, "Thiếu OPENAI_API_KEY trong file `.env`"

    try:
        import chromadb
        from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        client.get_collection(COLLECTION_NAME)
    except Exception:
        return False, "ChromaDB chưa có collection. Chạy: `python -m src.task4_chunking_indexing`"

    return True, "OK"


def get_index_stats() -> tuple[int, int]:
    """Trả về (số chunks, số documents) từ ChromaDB."""
    try:
        import chromadb
        from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION_NAME)
        n_chunks = col.count()
        meta = col.get(include=["metadatas"])
        sources = {
            (m or {}).get("source") or (m or {}).get("file") or (m or {}).get("path")
            for m in meta.get("metadatas", [])
        }
        sources.discard(None)
        return n_chunks, len(sources)
    except Exception:
        return 0, 0


def chat(
    query: str,
    history: list[dict] | None = None,
    top_k: int = 5,
    score_threshold: float = 0.3,
    use_follow_up: bool = True,
) -> dict:
    """
    Gọi RAG pipeline với optional conversation memory.

    Returns:
        {
            'query': str,              # câu hỏi gốc
            'effective_query': str,    # câu hỏi sau khi ghép history
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str,
        }
    """
    history = history or []
    effective_query = (
        build_follow_up_query(query, history) if use_follow_up and history else query
    )

    result = generate_with_citation(
        effective_query,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    return {
        "query": query,
        "effective_query": effective_query,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "retrieval_source": result.get("retrieval_source", "none"),
    }
