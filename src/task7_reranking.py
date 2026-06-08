"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Bi-encoder reranker: BAAI/bge-m3 (cùng Task 4, cosine query–doc)
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — dùng merge semantic + lexical ở Task 9

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Dùng cùng bi-encoder BAAI/bge-m3 từ Task 4 để re-score query–document.
# (Tránh tải thêm cross-encoder model; vẫn đo relevance query↔candidate trực tiếp.)


def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    from src.task4_chunking_indexing import _get_embedding_model

    model = _get_embedding_model()
    query_emb = model.encode(query, normalize_embeddings=True)
    doc_embs = model.encode(
        [c["content"][:2000] for c in candidates],
        normalize_embeddings=True,
    )
    scores = np.dot(doc_embs, query_emb)

    ranked = []
    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    limit = min(top_k, len(candidates))

    for _ in range(limit):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(
                    candidates[idx]["embedding"],
                    candidates[sel_idx]["embedding"],
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for i, idx in enumerate(selected):
        relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])
        max_sim = 0.0
        for prev_idx in selected[:i]:
            max_sim = max(
                max_sim,
                _cosine_sim(
                    candidates[idx]["embedding"],
                    candidates[prev_idx]["embedding"],
                ),
            )
        item = candidates[idx].copy()
        item["score"] = float(
            lambda_param * relevance - (1 - lambda_param) * max_sim
        )
        results.append(item)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        from src.task4_chunking_indexing import _get_embedding_model

        model = _get_embedding_model()
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()

        enriched = []
        for candidate in candidates:
            embedding = candidate.get("embedding")
            if embedding is None:
                embedding = model.encode(
                    candidate["content"],
                    normalize_embeddings=True,
                ).tolist()
            enriched.append({**candidate, "embedding": embedding})
        return rerank_mmr(query_embedding, enriched, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
