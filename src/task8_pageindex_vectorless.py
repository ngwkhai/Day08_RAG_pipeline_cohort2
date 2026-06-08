"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # TODO: Implement upload
    if not PAGEINDEX_API_KEY:
        print("Mock: API Key missing, skipping upload.")
        return

    try:
        from pageindex import PageIndexClient
        pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        
        landing_dir = Path(__file__).parent.parent / "data" / "landing" / "legal"
        pdf_files = list(landing_dir.rglob("*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                # Thư viện mới dùng hàm submit_document nhận vào đường dẫn file thay vì truyền content
                pi.submit_document(file_path=str(pdf_file))
                print(f"  [OK] Uploaded: {pdf_file.name}")
            except Exception as e:
                print(f"  [ERROR] Loi khi upload {pdf_file.name}: {e}")
    except Exception as e:
        print(f"Error uploading to pageindex: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # TODO: Implement PageIndex query
    if not PAGEINDEX_API_KEY:
        return [
            {
                "content": f"Mock vectorless fallback result for query: {query}",
                "score": 0.5,
                "metadata": {"source": "mock_pageindex"},
                "source": "pageindex"
            }
        ]

    try:
        from pageindex import PageIndexClient
        import time
        pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        
        docs = pi.list_documents(limit=50).get('documents', [])
        all_results = []
        
        for doc in docs:
            try:
                doc_id = doc['id']
                if doc.get('status') != 'completed':
                    continue
                    
                q_res = pi.submit_query(doc_id=doc_id, query=query, thinking=False)
                ret_id = q_res.get('retrieval_id')
                
                ret_data = None
                for _ in range(10): # đợi tối đa 20s cho mỗi doc
                    time.sleep(2)
                    status_data = pi.get_retrieval(ret_id)
                    if status_data.get('status') == 'completed':
                        ret_data = status_data
                        break
                
                if ret_data and 'retrieved_nodes' in ret_data:
                    for idx, node in enumerate(ret_data['retrieved_nodes']):
                        content = ""
                        if 'relevant_contents' in node and node['relevant_contents']:
                            try:
                                content = node['relevant_contents'][0][0].get('relevant_content', '')
                            except Exception:
                                pass
                        
                        simulated_score = max(0.1, 0.9 - (idx * 0.05))
                        
                        all_results.append({
                            "content": content,
                            "score": float(simulated_score),
                            "metadata": {"doc_id": doc_id},
                            "source": "pageindex"
                        })
            except Exception:
                continue
                
        all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        return all_results[:top_k] if all_results else [
            {"content": "Không tìm thấy kết quả từ PageIndex.", "score": 0.5, "metadata": {}, "source": "pageindex"}
        ]
    except Exception as e:
        print(f"Error querying pageindex: {e}")
        return [{"content": "Fallback mock", "score": 0.5, "metadata": {}, "source": "pageindex"}]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
