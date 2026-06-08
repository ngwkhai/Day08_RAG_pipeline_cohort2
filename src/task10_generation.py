"""
Task 10 — Generation Có Citation [Cập nhật định dạng Nguồn, Năm].

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation [Nguồn, Năm]
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

try:
    from task9_retrieval_pipeline import retrieve
except ImportError:
    from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.1


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là một trợ lý AI am hiểu về pháp luật và tin tức tại Việt Nam.
Hãy trả lời câu hỏi của người dùng một cách chi tiết và chính xác dựa CHỈ VÀO ngữ cảnh (Context) được cung cấp bên dưới.

Quy tắc bắt buộc:
1. Bất cứ khi nào bạn đưa ra một thông tin, số liệu, hoặc tuyên bố nào, bạn BẮT BUỘC phải trích dẫn nguồn ngay lập tức bằng cách sử dụng ngoặc vuông theo đúng định dạng [Nguồn, Năm].
   Ví dụ: "Tội tàng trữ ma túy có thể bị phạt tù [Luật Phòng Chống Ma Tuy, 2021]".
   (Lấy thông tin Nguồn và Năm tương ứng từ phần header của mỗi đoạn Context được cung cấp).
2. Chỉ sử dụng thông tin có trong Context. Tuyệt đối KHÔNG sử dụng kiến thức sẵn có của bạn để bịa thêm thông tin.
3. Nếu thông tin trong Context không đủ để trả lời câu hỏi hoặc lạc đề, hãy trả lời ĐÚNG nguyên văn câu này: 'Tôi không thể xác minh thông tin này từ nguồn hiện có'. Không đoán mò.
4. Trình bày câu trả lời rõ ràng, chia đoạn hoặc dùng gạch đầu dòng cho dễ đọc.
"""


# =============================================================================
# DOCUMENT REORDERING
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.
    Input order (by score descending):  [1, 2, 3, 4, 5]
    Output order:                       [1, 3, 5, 4, 2]
    """
    if not chunks:
        return []

    reordered = [None] * len(chunks)
    left = 0
    right = len(chunks) - 1

    for i in range(len(chunks)):
        if i % 2 == 0:
            reordered[left] = chunks[i]
            left += 1
        else:
            reordered[right] = chunks[i]
            right -= 1

    return reordered


# =============================================================================
# CONTEXT FORMATTING (Nâng cấp trích xuất năm)
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Tự động tách Tên file và Năm để mớm sẵn cho LLM cite chuẩn xác.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source_raw = metadata.get("source", f"Source_{i}")
        
        # Dùng Regex để tìm năm (các số bắt đầu bằng 19xx hoặc 20xx) trong tên file
        year_match = re.search(r'(19|20)\d{2}', source_raw)
        year = year_match.group() if year_match else "N/A"
        
        # Làm sạch tên nguồn: Bỏ đuôi file, thay dấu gạch ngang bằng khoảng trắng và viết hoa chữ đầu
        clean_source = source_raw.replace(".md", "").replace(".json", "").replace(".pdf", "")
        clean_source = clean_source.replace("-", " ").replace("_", " ").title()
        
        context_parts.append(
            f"[Nguồn: {clean_source} | Năm: {year}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION (Gemini)
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none"
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    from google import genai
    from google.genai import types

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return {
            "answer": "Lỗi: Không tìm thấy GEMINI_API_KEY trong file .env.",
            "sources": chunks,
            "retrieval_source": "error"
        }

    client = genai.Client(api_key=gemini_api_key)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=SYSTEM_PROMPT)]),
                types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
            ],
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        )
        answer = response.text
    except Exception as e:
        answer = f"Lỗi trong quá trình gọi Gemini LLM: {e}"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING TASK 10: RAG GENERATION VỚI GEMINI (CITATION CHUẨN)")
    print("=" * 70)
    
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
    ]

    for q in test_queries:
        print(f"\nHỎI: {q}")
        print("-" * 70)
        
        result = generate_with_citation(q)
        
        print(f"\nTRẢ LỜI:\n{result['answer']}")