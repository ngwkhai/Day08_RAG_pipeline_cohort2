"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # TODO: Implement semantic search
    import chromadb
    from sentence_transformers import SentenceTransformer
    from pathlib import Path
    
    STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
    db_path = STANDARDIZED_DIR.parent / "chroma_db"
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query, show_progress_bar=False).tolist()
    
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(name="DrugLawDocs")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        out = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                # Convert distance to score. Lower distance = higher score
                score = float(1.0 / (1.0 + distance))
                out.append({
                    "content": results["documents"][0][i],
                    "score": score,
                    "metadata": results["metadatas"][0][i]
                })
        return out
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
