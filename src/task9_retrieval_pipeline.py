"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.
"""

# Xử lý import linh hoạt để có thể chạy trực tiếp file hoặc chạy qua module
try:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search
except ImportError:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search

# =============================================================================
# CONFIGURATION
# =============================================================================

# Threshold 0.3 áp dụng khá tốt với logits của Cross-Encoder.
# Nếu best_score < 0.3, có nghĩa là tài liệu không thực sự liên quan.
SCORE_THRESHOLD = 0.3   
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.
    """
    print(f"\n[Pipeline] Bắt đầu tìm kiếm: '{query}'")
    
    # --- Step 1: Song song chạy Semantic + Lexical ---
    print("  ├─ Đang truy xuất bằng Vector (Semantic) & BM25 (Lexical)...")
    # Lấy số lượng gấp đôi top_k để có không gian lựa chọn khi gộp
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # --- Step 2: Merge bằng RRF ---
    print("  ├─ Đang gộp kết quả bằng Reciprocal Rank Fusion (RRF)...")
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    
    # Gắn nhãn nguồn là hybrid để phân biệt với kết quả từ pageindex
    for item in merged:
        item["source"] = "hybrid"

    # --- Step 3: Rerank (Chấm điểm lại) ---
    if use_reranking and merged:
        print(f"  ├─ Đang chấm điểm lại (Reranking) bằng {RERANK_METHOD}...")
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # --- Step 4: Check threshold → Fallback ---
    # Lấy điểm của văn bản top 1
    best_score = final_results[0]["score"] if final_results else -99.0

    if not final_results or best_score < score_threshold:
        print(f"  ⚠ CẢNH BÁO: Điểm cao nhất ({best_score:.3f}) < Ngưỡng ({score_threshold}).")
        print("  └─ KÍCH HOẠT FALLBACK: Truy vấn PageIndex bằng Gemini...")
        
        fallback_results = pageindex_search(query, top_k=top_k)
        
        if fallback_results:
            print("     ✓ Đã tìm thấy kết quả từ PageIndex!")
            return fallback_results
        else:
            print("     ✗ PageIndex cũng không tìm thấy kết quả. Trả về kết quả Hybrid tốt nhất có thể.")

    else:
        print(f"  └─ ✓ Tìm kiếm Hybrid thành công (Best Score: {best_score:.3f}).")

    return final_results[:top_k]


if __name__ == "__main__":
    # Test queries bao gồm 2 câu bình thường và 1 câu "lạc đề" để ép hệ thống dùng Fallback
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý",
        "Cách làm bánh xèo miền Tây ngon nhất"  # Câu này chắc chắn điểm < 0.3
    ]

    for q in test_queries:
        print("=" * 70)
        results = retrieve(q, top_k=3)
        print("-" * 70)
        for i, r in enumerate(results, 1):
            source_engine = r.get("source", "unknown").upper()
            file_name = r.get("metadata", {}).get("source", "unknown file")
            
            print(f"{i}. [{source_engine} | {file_name}] - Score: {r['score']:.3f}")
            print(f"   {r['content'][:150]}...\n")