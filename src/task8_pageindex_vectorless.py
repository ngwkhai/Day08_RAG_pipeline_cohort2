"""
Task 8 — PageIndex Vectorless RAG (Reasoning-based RAG) với Gemini.

Cài đặt:
    pip install pageindex google-genai python-dotenv markdown pdfkit
    (Yêu cầu hệ thống phải cài đặt wkhtmltopdf)
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Thư viện để convert MD sang PDF
import markdown
import pdfkit

# Tải biến môi trường
load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_TEMP_DIR = Path(__file__).parent.parent / "data" / "pdf_temp"
DOC_IDS_FILE = Path(__file__).parent / "pageindex_doc_ids.json"


def convert_md_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """
    Chuyển đổi file Markdown sang PDF có hỗ trợ Unicode Tiếng Việt.
    """
    try:
        # Đọc nội dung Markdown
        md_text = md_path.read_text(encoding="utf-8")
        
        # Chuyển MD sang HTML
        html_content = markdown.markdown(md_text)
        
        # Bọc HTML với meta UTF-8 để không bị lỗi font tiếng Việt
        html_with_meta = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Tùy chọn (Tắt log rác của pdfkit)
        options = {'quiet': ''}
        
        # Tạo PDF
        pdfkit.from_string(html_with_meta, str(pdf_path), options=options)
        return True
        
    except OSError as e:
        if "No wkhtmltopdf executable found" in str(e):
            print("\n⚠ LỖI THIẾU CÔNG CỤ: Hệ thống của bạn chưa cài đặt 'wkhtmltopdf'.")
            print("  - Nếu dùng Colab/Linux: Chạy lệnh `!apt-get install wkhtmltopdf`")
            print("  - Nếu dùng Windows: Tải và cài đặt từ https://wkhtmltopdf.org/downloads.html")
            raise e
        else:
            print(f"⚠ Lỗi khi convert file {md_path.name}: {e}")
            return False
    except Exception as e:
        print(f"⚠ Lỗi không xác định khi convert file {md_path.name}: {e}")
        return False


def upload_documents():
    """
    Convert markdown sang PDF, upload lên PageIndex và lưu lại doc_id.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ LỖI: Chưa có PAGEINDEX_API_KEY.")
        return

    try:
        from pageindex import PageIndexClient
        pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        
        md_files = list(STANDARDIZED_DIR.rglob("*.md"))
        if not md_files:
            print("⚠ Không tìm thấy file .md nào. Hãy chạy Task 3 trước!")
            return
            
        print(f"Bắt đầu xử lý và upload {len(md_files)} tài liệu...")
        
        # Tạo thư mục lưu PDF tạm thời nếu chưa có
        PDF_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        doc_ids_map = {}
        
        for md_file in md_files:
            pdf_path = PDF_TEMP_DIR / f"{md_file.stem}.pdf"
            
            print(f"\n  Đang chuyển đổi: {md_file.name} -> PDF...")
            if convert_md_to_pdf(md_file, pdf_path):
                print(f"  Đang upload file PDF...")
                
                # Upload file PDF thay vì file Markdown
                response = pi_client.submit_document(str(pdf_path))
                doc_id = response.get("doc_id")
                
                if doc_id:
                    doc_type = "legal" if "legal" in md_file.parts else "news"
                    # Vẫn lưu ánh xạ với tên file .md gốc để đồng bộ nguồn với các Task trước
                    doc_ids_map[md_file.name] = {
                        "doc_id": doc_id,
                        "type": doc_type
                    }
                    print(f"    ✓ Thành công (doc_id: {doc_id})")
                
        # Lưu dữ liệu quản lý
        DOC_IDS_FILE.write_text(json.dumps(doc_ids_map, indent=2), encoding="utf-8")
        print(f"\n✓ Hoàn tất! Đã lưu ánh xạ ID vào {DOC_IDS_FILE.name}")

    except Exception as e:
        print(f"\n⚠ Dừng upload do lỗi: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng LLM Reasoning (Gemini) trên PageIndex Tree.
    """
    if not PAGEINDEX_API_KEY or not GEMINI_API_KEY:
        print("⚠ Cần có cả PAGEINDEX_API_KEY và GEMINI_API_KEY trong .env để chạy search.")
        return []

    if not DOC_IDS_FILE.exists():
        print("⚠ Chưa có dữ liệu tài liệu. Hãy gọi hàm upload_documents() trước!")
        return []

    try:
        from pageindex import PageIndexClient
        import pageindex.utils as utils
        from google import genai
        from google.genai import types

        doc_ids_map = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
        pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        MODEL_ID = "gemini-2.5-flash" 

        results = []

        for doc_name, doc_info in doc_ids_map.items():
            if len(results) >= top_k:
                break
                
            doc_id = doc_info["doc_id"]
            if not pi_client.is_retrieval_ready(doc_id):
                continue
                
            tree = pi_client.get_tree(doc_id, node_summary=True)['result']
            tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

            search_prompt = f"""
            You are a helpful assistant. You are given a question and a document tree structure.
            Find all nodes that are likely to contain the answer to the question.
            
            Question: {query}
            Document Name: {doc_name}
            Document Tree: {json.dumps(tree_without_text)}
            """

            response_schema = {
                "type": "OBJECT",
                "properties": {
                    "node_list": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    }
                },
                "required": ["node_list"]
            }

            response = gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=search_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.0,
                ),
            )
            
            llm_decision = json.loads(response.text)
            selected_nodes = llm_decision.get("node_list", [])

            if selected_nodes:
                node_map = utils.create_node_mapping(tree)
                for n_id in selected_nodes:
                    if n_id in node_map and node_map[n_id].get("text"):
                        results.append({
                            "content": node_map[n_id]["text"],
                            "score": 1.0, 
                            "metadata": {
                                "source": doc_name,
                                "type": doc_info["type"],
                                "node_id": n_id
                            },
                            "source": "pageindex"
                        })
                        if len(results) >= top_k:
                            break

        return results[:top_k]

    except Exception as e:
        print(f"⚠ Lỗi truy xuất PageIndex (Gemini): {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY or not GEMINI_API_KEY:
        print("=" * 50)
        print("⚠ YÊU CẦU API KEY:")
        print("Hãy đảm bảo file .env của bạn có:")
        print("PAGEINDEX_API_KEY=...")
        print("GEMINI_API_KEY=...")
        print("=" * 50)
    else:
        print("Bắt đầu chạy Task 8 (Gemini + PDF Conversion)...")
        
        # Mở comment dòng dưới ĐỂ CHẠY UPLOAD, sau khi tải xong thì comment lại.
        upload_documents()

        print("\nTest query:")
        search_results = pageindex_search("hình phạt tàng trữ trái phép chất ma tuý", top_k=2)
        
        if search_results:
            for i, r in enumerate(search_results, 1):
                print(f"\n--- Kết quả {i} [Từ: {r['metadata']['source']}] ---")
                print(f"{r['content'][:200]}...")
        else:
            print("Không tìm thấy kết quả phù hợp.")