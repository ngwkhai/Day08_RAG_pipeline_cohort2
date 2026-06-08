"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Configure, Property, DataType

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Chunking strategy
# Lý do: RecursiveCharacterTextSplitter an toàn và giữ được cấu trúc đoạn văn tốt. 
# Với size 500 và overlap 50, nó đủ lớn để chứa 1-2 Điều khoản luật trọn vẹn 
# nhưng không quá dài khiến embedding bị loãng (lost focus).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Embedding model
# Lý do: 'BAAI/bge-m3' là mô hình State-of-the-Art (SOTA) cho đa ngôn ngữ, 
# vượt trội trong việc hiểu ngữ nghĩa tiếng Việt so với all-MiniLM. 
# Dimension 1024 mang lại không gian biểu diễn chi tiết hơn.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Vector store
# Lý do: Weaviate được thiết kế native cho Hybrid Search. Ở các task sau,
# chúng ta có thể kết hợp Dense và Sparse search rất mượt mà trên cùng 1 DB.
VECTOR_STORE = "weaviate"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    
    if not STANDARDIZED_DIR.exists():
        print(f"⚠ Thư mục {STANDARDIZED_DIR} không tồn tại. Chạy Task 3 trước!")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        
        # Xác định type dựa vào thư mục cha (legal hoặc news)
        doc_type = "legal" if "legal" in md_file.parts else "news"
        
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
        
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents sử dụng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Bổ sung "Điều" vào separators để cắt đẹp hơn với văn bản Luật
        separators=["\n\n", "\n", "Điều ", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
            
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng BAAI/bge-m3.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # Khởi tạo model (nếu chạy lần đầu sẽ tải weights từ HuggingFace)
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [c["content"] for c in chunks]
    print(f"  Đang nhúng {len(texts)} chunks (có thể mất chút thời gian)...")
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
        
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào Weaviate.
    """
    if not chunks:
        return

    try:
        # Kết nối tới Weaviate (Yêu cầu phải chạy Weaviate server local)
        print("  Đang khởi động Weaviate Embedded (lần đầu có thể mất chút thời gian tải file)...")
        client = weaviate.connect_to_embedded()
        
        # Tạo collection mới (Xóa cái cũ nếu đã tồn tại để tránh rác data khi test)
        collection_name = "DrugLawDocs"
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(), # Do chúng ta tự embed ở hàm trên
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )

        # Batch insert để tối ưu tốc độ
        print(f"  Đang đẩy {len(chunks)} chunks lên Weaviate...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                    vector=chunk["embedding"]
                )
                
        print("  ✓ Indexed thành công vào Weaviate!")
        client.close()
        
    except weaviate.exceptions.WeaviateConnectionError:
        print("\n  ✗ LỖI KẾT NỐI WEAVIATE!")
        print("  ⚠ Đảm bảo bạn đã khởi động Weaviate server.")
        print("  Gợi ý chạy bằng Docker:")
        print("  docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.24.4\n")
    except Exception as e:
        print(f"\n  ✗ Lỗi không xác định khi lưu vào vector store: {e}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    
    if not docs:
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)


if __name__ == "__main__":
    run_pipeline()