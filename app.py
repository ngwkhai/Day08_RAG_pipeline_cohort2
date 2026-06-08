"""
Streamlit RAG chatbot for the group project.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.task10_generation import generate_with_citation  # noqa: E402


st.set_page_config(
    page_title="DrugLaw RAG Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
    .source-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        background: #fafafa;
    }
    .source-meta {
        color: #4b5563;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "source_history" not in st.session_state:
        st.session_state.source_history = []


def build_followup_query(question: str, max_turns: int = 4) -> str:
    """Add recent chat history so follow-up questions have context."""
    recent_messages = st.session_state.messages[-max_turns * 2 :]
    if not recent_messages:
        return question

    history_lines = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("\n", " ").strip()
        history_lines.append(f"{role}: {content[:700]}")

    history = "\n".join(history_lines)
    return (
        "Conversation history:\n"
        f"{history}\n\n"
        "Current question:\n"
        f"{question}"
    )


def source_title(source: dict, index: int) -> str:
    metadata = source.get("metadata", {}) or {}
    name = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("url")
        or f"Source {index}"
    )
    score = source.get("score")
    source_kind = source.get("source", "unknown")
    if isinstance(score, (int, float)):
        return f"{index}. {name} | {source_kind} | score {score:.3f}"
    return f"{index}. {name} | {source_kind}"


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No source chunks were returned.")
        return

    for i, source in enumerate(sources, 1):
        metadata = source.get("metadata", {}) or {}
        content = source.get("content", "")
        with st.expander(source_title(source, i), expanded=i == 1):
            cols = st.columns([1, 1, 1])
            cols[0].caption(f"Type: {metadata.get('type') or metadata.get('doc_type') or 'unknown'}")
            cols[1].caption(f"Retrieval: {source.get('source', 'unknown')}")
            cols[2].caption(f"Chunk: {metadata.get('chunk_index', 'n/a')}")
            if metadata:
                st.json(metadata, expanded=False)
            st.markdown(content[:2500] or "_Empty source content_")


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.source_history = []


init_state()

with st.sidebar:
    st.title("RAG Controls")
    top_k = st.slider("Source chunks", min_value=3, max_value=10, value=5, step=1)
    use_memory = st.toggle("Follow-up memory", value=True)

    st.divider()
    st.button("Clear chat", use_container_width=True, on_click=clear_chat)

    st.divider()
    st.caption("Pipeline")
    st.write("Task 9 retrieval -> Task 10 generation")
    st.write("Hybrid retrieval with PageIndex fallback")


st.title("DrugLaw RAG Chatbot")
st.caption("Ask questions about Vietnamese drug law and related news. Answers include citations and source chunks.")

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            sources = st.session_state.source_history[idx] if idx < len(st.session_state.source_history) else []
            if sources:
                render_sources(sources)


prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.source_history.append([])

    with st.chat_message("user"):
        st.markdown(prompt)

    query = build_followup_query(prompt) if use_memory else prompt

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer..."):
            try:
                result = generate_with_citation(query, top_k=top_k)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                retrieval_source = result.get("retrieval_source", "unknown")
            except Exception as exc:
                answer = f"Pipeline error: {exc}"
                sources = []
                retrieval_source = "error"

        st.markdown(answer)
        st.caption(f"Retrieval source: {retrieval_source}")
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.source_history.append(sources)
