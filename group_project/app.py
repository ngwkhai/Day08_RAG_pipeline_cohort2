"""
RAG Chatbot — Pháp luật ma tuý & Tin tức liên quan.

Chạy:
    streamlit run group_project/app.py
"""

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from group_project.rag_service import chat, check_pipeline_ready, get_index_stats

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Pháp Luật Ma Tuý",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(15, 23, 42, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.88;
        font-size: 0.95rem;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-hybrid {
        background: #dbeafe;
        color: #1d4ed8;
    }
    .badge-pageindex {
        background: #fef3c7;
        color: #b45309;
    }
    .badge-none {
        background: #f1f5f9;
        color: #64748b;
    }

    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0d9488;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .source-card .source-title {
        font-weight: 600;
        color: #0f172a;
        font-size: 0.88rem;
        margin-bottom: 0.35rem;
    }
    .source-card .source-meta {
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .source-card .source-snippet {
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.55;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    .example-btn button {
        width: 100%;
        text-align: left !important;
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        background: white !important;
        color: #334155 !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 0.75rem !important;
        transition: all 0.15s ease !important;
    }
    .example-btn button:hover {
        border-color: #0d9488 !important;
        color: #0d9488 !important;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.15) !important;
    }

    .stat-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .stat-box .stat-num {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0d9488;
    }
    .stat-box .stat-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
    }

    .welcome-box {
        background: #f0fdfa;
        border: 1px dashed #99f6e4;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        color: #134e4a;
    }
    .welcome-box h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    .welcome-box ul {
        margin: 0;
        padding-left: 1.2rem;
        font-size: 0.88rem;
        line-height: 1.7;
    }
</style>
""",
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
    "Luật Phòng chống ma tuý quy định những hình thức cai nghiện nào?",
    "Ca sĩ Long Nhật bị xử lý thế nào trong vụ liên quan ma tuý?",
    "Ma túy trong lối sống showbiz được báo chí phân tích ra sao?",
    "Quy trình xác định tình trạng nghiện ma tuý theo quy định hiện hành?",
]

SOURCE_BADGE = {
    "hybrid": ("badge badge-hybrid", "Hybrid Search"),
    "pageindex": ("badge badge-pageindex", "PageIndex"),
    "none": ("badge badge-none", "Không có nguồn"),
}


def init_session_state():
    defaults = {
        "messages": [],
        "pending_query": None,
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_source_badge(source: str):
    css_class, label = SOURCE_BADGE.get(source, SOURCE_BADGE["none"])
    st.markdown(f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True)


def render_sources(sources: list[dict]):
    if not sources:
        st.caption("Không có nguồn tham khảo được truy xuất.")
        return

    for i, src in enumerate(sources, 1):
        meta = src.get("metadata") or {}
        name = meta.get("source") or meta.get("file") or meta.get("path") or f"Chunk {i}"
        doc_type = meta.get("type") or meta.get("doc_type") or "unknown"
        score = src.get("score", 0)
        snippet = (src.get("content") or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"

        st.markdown(
            f"""
<div class="source-card">
  <div class="source-title">📄 {name}</div>
  <div class="source-meta">Loại: {doc_type} · Score: {score:.3f}</div>
  <div class="source-snippet">{snippet}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_assistant_turn(turn: dict):
    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.markdown(turn["answer"])
    with col_b:
        render_source_badge(turn.get("retrieval_source", "none"))
    with st.expander(f"📚 Nguồn tham khảo ({len(turn.get('sources', []))} chunks)", expanded=False):
        render_sources(turn.get("sources", []))


def process_query(user_query: str, top_k: int, score_threshold: float, use_follow_up: bool):
    """Gọi pipeline và lưu kết quả vào session — không render trực tiếp."""
    history_before = st.session_state.messages[:-1]
    try:
        result = chat(
            query=user_query,
            history=history_before,
            top_k=top_k,
            score_threshold=score_threshold,
            use_follow_up=use_follow_up,
        )
        st.session_state.messages[-1]["answer"] = result["answer"]
        st.session_state.messages[-1]["sources"] = result.get("sources", [])
        st.session_state.messages[-1]["retrieval_source"] = result.get("retrieval_source", "none")
        st.session_state.messages[-1]["status"] = "done"
    except Exception as exc:
        st.session_state.messages[-1]["answer"] = f"⚠️ Lỗi pipeline: {exc}"
        st.session_state.messages[-1]["sources"] = []
        st.session_state.messages[-1]["retrieval_source"] = "none"
        st.session_state.messages[-1]["status"] = "error"


def main():
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Cấu hình")
        top_k = st.slider("Số chunks retrieve (top_k)", 3, 10, 5)
        score_threshold = st.slider(
            "Ngưỡng fallback PageIndex",
            0.0, 0.9, 0.3, 0.05,
            help="Nếu điểm hybrid thấp hơn ngưỡng này → fallback sang PageIndex",
        )
        use_follow_up = st.toggle("Conversation memory (follow-up)", value=True)

        st.markdown("---")
        st.markdown("### 💡 Câu hỏi mẫu")
        for q in EXAMPLE_QUESTIONS:
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(q, key=f"ex_{hash(q)}"):
                st.session_state.pending_query = q
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

        st.markdown("---")
        st.markdown("#### Pipeline")
        st.caption(
            "Task 9 retrieve → Task 10 generate\n\n"
            "Semantic + BM25 → RRF → Rerank\n\n"
            "Fallback PageIndex khi score thấp"
        )

    ready, msg = check_pipeline_ready()
    n_chunks, n_docs = get_index_stats()

    if not ready:
        st.warning(msg)

    # Header
    st.markdown(
        """
<div class="main-header">
  <h1>⚖️ RAG Chatbot — Pháp Luật & Tin Tức Ma Tuý</h1>
  <p>Trả lời có trích dẫn nguồn · Hybrid retrieval · Hỗ trợ câu hỏi follow-up</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-num">{len(st.session_state.messages)}</div>'
            f'<div class="stat-label">Lượt hội thoại</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-num">{n_chunks}</div>'
            f'<div class="stat-label">Chunks indexed</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-box"><div class="stat-num">{n_docs}</div>'
            f'<div class="stat-label">Documents</div></div>',
            unsafe_allow_html=True,
        )

    if not st.session_state.messages:
        st.markdown(
            """
<div class="welcome-box">
  <h3>👋 Chào mừng — Hỏi về pháp luật ma tuý hoặc tin tức liên quan</h3>
  <ul>
    <li>Mỗi câu trả lời kèm <strong>citation</strong> từ văn bản pháp luật hoặc bài báo</li>
    <li>Bấm <strong>Câu hỏi mẫu</strong> ở sidebar hoặc gõ câu hỏi follow-up (vd: "còn hình phạt thì sao?")</li>
    <li>Mở <strong>Nguồn tham khảo</strong> để xem chunks đã retrieve</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    # Chat history — single source of truth
    for turn in st.session_state.messages:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(turn["query"])
        with st.chat_message("assistant", avatar="⚖️"):
            if turn.get("status") == "pending":
                with st.spinner("Đang truy xuất tài liệu và soạn câu trả lời…"):
                    process_query(turn["query"], top_k, score_threshold, use_follow_up)
                render_assistant_turn(turn)
            else:
                render_assistant_turn(turn)

    # Sidebar example → enqueue
    if st.session_state.pending_query:
        q = st.session_state.pending_query
        st.session_state.pending_query = None
        st.session_state.messages.append({
            "query": q,
            "answer": "",
            "sources": [],
            "retrieval_source": "none",
            "status": "pending",
        })
        st.rerun()

    # Chat input → enqueue
    if user_input := st.chat_input("Nhập câu hỏi về pháp luật ma tuý hoặc tin tức liên quan…"):
        st.session_state.messages.append({
            "query": user_input,
            "answer": "",
            "sources": [],
            "retrieval_source": "none",
            "status": "pending",
        })
        st.rerun()


if __name__ == "__main__":
    main()
