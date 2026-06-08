"""
Task 7 — Reranking Module.

Implement 2 phương pháp chính để sử dụng trong Hybrid Search:
    1. RRF (Reciprocal Rank Fusion): Dùng để gộp kết quả Dense (Task 5) và Sparse (Task 6).
    2. Cross-encoder: Dùng để chấm điểm lại chính xác độ liên quan.

Cài đặt (nếu dùng Cross-encoder):
    pip install sentence-transformers
"""

from typing import Optional

# Khởi tạo CrossEncoder ở global scope để tối ưu
try:
    from sentence_transformers import CrossEncoder
    print("Loading Cross-Encoder model 'BAAI/bge-reranker-base'...")
    # Dùng bản base để nhẹ RAM nhưng vẫn vượt trội về ngữ nghĩa đa ngôn ngữ
    cross_encoder_model = CrossEncoder('BAAI/bge-reranker-base')
    print("✓ Cross-Encoder loaded.")
except ImportError:
    print("⚠ Chưa cài sentence-transformers. Cross-encoder sẽ không hoạt động.")
    cross_encoder_model = None


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model. Thích hợp để "chuốt" lại 
    kết quả cuối cùng trước khi đưa vào LLM.
    """
    if not candidates:
        return []
        
    if cross_encoder_model is None:
        raise RuntimeError("Model Cross-Encoder chưa được khởi tạo.")

    # Tạo các cặp (Câu hỏi, Văn bản) để mô hình đọc chéo
    pairs = [[query, doc["content"]] for doc in candidates]
    
    # Predict trả về mảng điểm số (điểm có thể âm hoặc dương, càng cao càng tốt)
    scores = cross_encoder_model.predict(pairs)

    # Gán điểm mới và sắp xếp lại
    reranked_candidates = []
    for doc, score in zip(candidates, scores):
        doc_copy = doc.copy()
        doc_copy["score"] = float(score)
        # Đánh dấu nguồn điểm để dễ debug
        doc_copy["metadata"] = {**doc_copy["metadata"], "rerank_method": "cross_encoder"}
        reranked_candidates.append(doc_copy)

    # Sort descending
    reranked_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    return reranked_candidates[:top_k]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker (Dense + Sparse).
    Thuật toán: RRF(d) = Σ 1 / (k + rank_r(d))
    
    Giải thích: 
    - Văn bản ở Top 1 được cộng điểm cao nhất (1/61)
    - Nếu 1 văn bản xuất hiện ở Top của cả 2 thuật toán, nó sẽ được cộng dồn điểm.
    """
    if not ranked_lists:
        return []

    rrf_scores = {}      # content -> RRF score
    content_map = {}     # content -> dict lưu trữ document

    for ranked_list in ranked_lists:
        # enumerate(..., 1) để rank bắt đầu từ 1
        for rank, item in enumerate(ranked_list, 1):
            content = item["content"]
            
            # Tính điểm RRF cho vị trí hiện tại
            score = 1.0 / (k + rank)
            
            if content not in rrf_scores:
                rrf_scores[content] = 0.0
                content_map[content] = item
                
            rrf_scores[content] += score

    # Sắp xếp document theo điểm RRF giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        item["metadata"] = {**item.metadata, "rerank_method": "rrf"} if hasattr(item, "metadata") else {"rerank_method": "rrf"}
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
    ranked_lists: Optional[list[list[dict]]] = None
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        if not ranked_lists:
            raise ValueError("Method 'rrf' yêu cầu truyền vào 'ranked_lists'.")
        return rerank_rrf(ranked_lists, top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Reranking Module")
    print("=" * 50)
    
    # 1. Test RRF (Gộp 2 danh sách giả lập)
    list_vector = [
        {"content": "Tội tàng trữ ma tuý theo Điều 248.", "score": 0.9, "metadata": {"source": "vector"}},
        {"content": "Cai nghiện ma tuý tại nhà.", "score": 0.8, "metadata": {"source": "vector"}},
    ]
    list_bm25 = [
        {"content": "Nghệ sĩ sử dụng ma tuý.", "score": 15.4, "metadata": {"source": "bm25"}},
        {"content": "Tội tàng trữ ma tuý theo Điều 248.", "score": 12.1, "metadata": {"source": "bm25"}},
    ]
    
    print("\n--- Test RRF (Gộp kết quả Vector & BM25) ---")
    rrf_results = rerank("", candidates=[], top_k=3, method="rrf", ranked_lists=[list_vector, list_bm25])
    for i, r in enumerate(rrf_results, 1):
        print(f"[{r['score']:.4f}] {r['content']}")

    # 2. Test Cross-Encoder
    if cross_encoder_model:
        print("\n--- Test Cross-Encoder ---")
        query = "Hình phạt tàng trữ trái phép chất ma tuý"
        candidates = [
            {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý trong quán bar.", "score": 0, "metadata": {}},
            {"content": "Người nào tàng trữ trái phép chất ma túy thì bị phạt tù từ 01 năm đến 05 năm.", "score": 0, "metadata": {}},
            {"content": "Ma tuý là chất gây nghiện nguy hiểm.", "score": 0, "metadata": {}},
        ]
        
        ce_results = rerank(query, candidates, top_k=2, method="cross_encoder")
        for i, r in enumerate(ce_results, 1):
            print(f"[{r['score']:.4f}] {r['content']}")