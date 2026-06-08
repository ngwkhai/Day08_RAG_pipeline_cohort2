"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from src.task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random, phù hợp câu trả lời factual
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo, giảm hallucination
TEMPERATURE = 0.3

LLM_MODEL = "gpt-4o-mini"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    # Chỉ số chẵn (0, 2, 4, ...) → đầu prompt; chỉ số lẻ đảo ngược → cuối prompt
    reordered = [chunks[i] for i in range(0, len(chunks), 2)]
    odd_start = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(odd_start, 0, -2):
        reordered.append(chunks[i])
    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def _source_label(metadata: dict) -> str:
    """Tạo nhãn nguồn dễ cite cho LLM."""
    source = metadata.get("source") or metadata.get("file") or "Unknown"
    doc_type = metadata.get("type") or metadata.get("doc_type") or ""
    path = metadata.get("path") or ""

    label = source
    if doc_type:
        label = f"{label} ({doc_type})"
    if path and path != source:
        label = f"{label} | {path}"
    return label


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "(Không có ngữ cảnh liên quan)"

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = _source_label(metadata)
        score = chunk.get("score", 0)
        context_parts.append(
            f"[Document {i} | Source: {source} | Score: {score:.3f}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def _call_llm(system_prompt: str, user_message: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-xxx":
        raise ValueError(
            "OPENAI_API_KEY chưa được cấu hình trong .env"
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    score_threshold: float | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks retrieve

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    from src.task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve

    threshold = SCORE_THRESHOLD if score_threshold is None else score_threshold
    chunks = retrieve(query, top_k=top_k, score_threshold=threshold)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = (
        f"Context:\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        "Hãy trả lời bằng tiếng Việt, mỗi nhận định phải có citation [Nguồn, Năm]."
    )

    answer = _call_llm(SYSTEM_PROMPT, user_message)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(
            f"\n[Sources: {len(result['sources'])} chunks | "
            f"via {result['retrieval_source']}]"
        )
