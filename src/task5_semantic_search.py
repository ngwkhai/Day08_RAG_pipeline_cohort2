"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chromadb

from src.task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    _get_embedding_model,
)

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


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
    if top_k <= 0:
        return []

    model = _get_embedding_model()
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    collection = _get_collection()
    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        # Chroma cosine distance = 1 - cosine_similarity (với vector đã normalize)
        score = 1.0 - distance
        output.append({
            "content": doc,
            "score": float(score),
            "metadata": {
                "source": meta.get("source", ""),
                "doc_type": meta.get("doc_type", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "path": meta.get("path", ""),
            },
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata']['source']}) {r['content'][:100]}...")
