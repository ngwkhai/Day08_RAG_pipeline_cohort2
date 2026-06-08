"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import weaviate
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer

# Load model một lần duy nhất ở global scope để tối ưu tốc độ 
# (tránh việc mỗi lần gọi hàm lại phải load lại model vài GB vào RAM)
try:
    print("Loading embedding model 'BAAI/bge-m3'...")
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    print("✓ Model loaded successfully.")
except Exception as e:
    print(f"⚠ Lỗi load model: {e}")
    embedding_model = None


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity qua Weaviate.

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
    if embedding_model is None:
        raise RuntimeError("Embedding model chưa được khởi tạo!")

    # Bước 1: Nhúng (embed) câu truy vấn thành vector
    query_embedding = embedding_model.encode(query).tolist()

    results = []
    client = None

    try:
        # Bước 2: Kết nối tới Weaviate và truy vấn
        client = weaviate.connect_to_embedded()
        
        collection = client.collections.get("DrugLawDocs")
        collection = client.collections.get("DrugLawDocs")

        # Thực hiện tìm kiếm vector (near_vector)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )

        # Bước 3: Format kết quả trả về
        for obj in response.objects:
            # Weaviate trả về 'distance'. 
            # Với Cosine distance, Similarity = 1 - Distance
            score = 1.0 - obj.metadata.distance

            results.append({
                "content": obj.properties["content"],
                "score": float(score),
                "metadata": {
                    "source": obj.properties.get("source", "unknown"),
                    "type": obj.properties.get("doc_type", "unknown"),
                    "chunk_index": obj.properties.get("chunk_index", -1)
                }
            })

    except weaviate.exceptions.WeaviateConnectionError:
        print("\n⚠ LỖI: Không thể kết nối tới Weaviate. Hãy chắc chắn Docker Weaviate đang chạy!")
    except Exception as e:
        print(f"\n⚠ LỖI trong quá trình Semantic Search: {e}")
    finally:
        # Đảm bảo đóng kết nối an toàn
        if client is not None:
            client.close()

    return results


if __name__ == "__main__":
    # Test script
    print("=" * 50)
    print("Testing Semantic Search")
    print("=" * 50)
    
    test_query = "hình phạt cho tội tàng trữ ma tuý"
    print(f"\nQuery: '{test_query}'\n")
    
    search_results = semantic_search(test_query, top_k=3)
    
    if search_results:
        for i, r in enumerate(search_results, 1):
            print(f"Kết quả {i} | Score: {r['score']:.4f} | Nguồn: {r['metadata']['source']}")
            print(f"Trích xuất: {r['content'][:150]}...\n")
    else:
        print("Không tìm thấy kết quả hoặc có lỗi xảy ra.")