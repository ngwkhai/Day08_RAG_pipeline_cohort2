"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Corpus = chunks từ Task 4 (cùng granularity với semantic search)
CORPUS: list[dict] = []
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản: lowercase + split theo khoảng trắng."""
    return text.lower().split()


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global CORPUS, _bm25
    CORPUS = corpus
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25


def _ensure_index():
    """Lazy-load corpus và BM25 index từ Task 4 chunks."""
    global _bm25
    if _bm25 is not None:
        return

    from src.task4_chunking_indexing import chunk_documents, load_documents

    docs = load_documents()
    if not docs:
        build_bm25_index([])
        return

    build_bm25_index(chunk_documents(docs))


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    _ensure_index()
    if not CORPUS or _bm25 is None:
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        if len(results) >= top_k:
            break
        score = float(scores[idx])
        if score <= 0:
            break
        results.append({
            "content": CORPUS[idx]["content"],
            "score": score,
            "metadata": CORPUS[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata'].get('source', '')}) {r['content'][:100]}...")
