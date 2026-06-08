"""
Task 6 — Lexical Search Module (BM25).

Cài đặt:
    pip install rank-bm25 numpy
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi

# Tái sử dụng logic đọc và cắt chunk từ Task 4 để đảm bảo đồng nhất dữ liệu
try:
    from task4_chunking_indexing import load_documents, chunk_documents
except ImportError:
    from .task4_chunking_indexing import load_documents, chunk_documents

# Biến toàn cục lưu trữ BM25 index và Corpus gốc
CORPUS: list[dict] = []
bm25_index = None


def tokenize_vietnamese(text: str) -> list[str]:
    """
    Tokenize văn bản tiếng Việt đơn giản bằng Regex.
    Chuyển về chữ thường và tách từ dựa trên các ký tự chữ/số.
    """
    text = text.lower()
    # Lấy danh sách các từ (bao gồm cả chữ cái tiếng Việt có dấu và số)
    return re.findall(r'\w+', text)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.
    
    Thuật toán khởi tạo một Inverted Index ngầm định tính toán TF-IDF
    có điều chỉnh (chuẩn hóa độ dài văn bản).
    """
    global bm25_index, CORPUS
    CORPUS = corpus
    
    print(f"  Đang tokenize {len(corpus)} chunks để xây dựng BM25 Index...")
    tokenized_corpus = [tokenize_vietnamese(doc["content"]) for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("  ✓ Xây dựng BM25 Index thành công!")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng thuật toán BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
        Sorted by score descending.
    """
    if bm25_index is None or not CORPUS:
        raise ValueError("BM25 Index chưa được khởi tạo. Hệ thống chưa sẵn sàng.")

    tokenized_query = tokenize_vietnamese(query)
    
    # Lấy mảng điểm số của tất cả document so với query (O(N) time complexity)
    scores = bm25_index.get_scores(tokenized_query)

    # Lấy ra các index có điểm cao nhất (tối ưu bằng np.argsort)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0:  # Loại bỏ các document hoàn toàn không chứa từ khóa nào
            results.append({
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx]["metadata"]
            })
            
    return results


# =============================================================================
# AUTO-INITIALIZATION THÔNG MINH
# =============================================================================
# Tự động tải dữ liệu và xây index khi module này được import vào Task 9.
# Điều này giúp server không bị nghẽn do phải tính lại index cho mỗi truy vấn.

print("\nKhởi tạo Lexical Search Module...")
try:
    _docs = load_documents()
    if _docs:
        _chunks = chunk_documents(_docs)
        build_bm25_index(_chunks)
    else:
        print("  ⚠ Không tìm thấy tài liệu gốc. Bỏ qua khởi tạo BM25.")
except Exception as e:
    print(f"  ⚠ Lỗi khi khởi tạo corpus BM25: {e}")


if __name__ == "__main__":
    # Test script
    print("=" * 50)
    print("Testing Lexical Search (BM25)")
    print("=" * 50)
    
    test_query = "Điều 248 tàng trữ trái phép chất ma tuý"
    print(f"\nQuery: '{test_query}'\n")
    
    search_results = lexical_search(test_query, top_k=3)
    
    if search_results:
        for i, r in enumerate(search_results, 1):
            print(f"Kết quả {i} | Score: {r['score']:.4f} | Nguồn: {r['metadata']['source']}")
            print(f"Trích xuất: {r['content'][:150]}...\n")
    else:
        print("Không tìm thấy kết quả phù hợp (hoặc từ khóa không tồn tại trong corpus).")