"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search

# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"
CANDIDATE_MULTIPLIER = 2  # Lấy top_k * 2 candidates trước khi merge/rerank


def _tag_hybrid(results: list[dict]) -> list[dict]:
    tagged = []
    for item in results:
        row = item.copy()
        row["source"] = "hybrid"
        tagged.append(row)
    return tagged


def _hybrid_retrieve(
    query: str,
    top_k: int,
    use_reranking: bool,
) -> list[dict]:
    """Chạy semantic + lexical, merge RRF, rerank."""
    candidate_k = max(top_k * CANDIDATE_MULTIPLIER, top_k)

    dense_results = semantic_search(query, top_k=candidate_k)
    sparse_results = lexical_search(query, top_k=candidate_k)

    if not dense_results and not sparse_results:
        return []

    merged = rerank_rrf(
        [r for r in (dense_results, sparse_results) if r],
        top_k=candidate_k,
    )
    merged = _tag_hybrid(merged)

    if use_reranking and merged:
        return rerank(query, merged, top_k=top_k, method=RERANK_METHOD)

    return merged[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    hybrid_results = _hybrid_retrieve(query, top_k, use_reranking)
    best_score = hybrid_results[0]["score"] if hybrid_results else 0.0

    if hybrid_results and best_score >= score_threshold:
        return hybrid_results[:top_k]

    # Fallback PageIndex khi không có kết quả hoặc score thấp
    try:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]
    except Exception as exc:
        print(f"  ⚠ PageIndex fallback thất bại: {exc}")

    return hybrid_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
