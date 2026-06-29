"""
Personal Study Coach — Streamlit frontend.

Connects to the FastAPI backend at API_BASE_URL (default: http://localhost:10000).
Run locally:
    Terminal 1 ▸ uv run uvicorn main:app --port 10000
    Terminal 2 ▸ uv run streamlit run streamlit_app.py
"""

import uuid

import streamlit as st

from frontend import agent_tab, chat_tab, metrics_tab, quiz_tab
from frontend.api_client import APIError, get_documents, health, upload_pdf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Study Coach",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "session_id":    lambda: str(uuid.uuid4())[:8],
    "messages":      list,
    "quiz":          lambda: None,
    "quiz_result":   lambda: None,
    "upload_status": lambda: None,   # ("ok"|"err", message)
    "study_plan":    lambda: None,   # agent result dict
}
for key, factory in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = factory()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📘 Personal Study Coach")
    st.caption("RAG · Hybrid Search · Cross-encoder Reranking")
    st.divider()

    # -- Connection status
    try:
        h = health()
        if h.get("status") == "healthy":
            st.success("API connected", icon="✅")
        else:
            st.warning("API degraded — check server logs", icon="⚠️")
    except APIError as e:
        st.error("API offline", icon="🔴")
        with st.expander("How to start the server"):
            st.code("uv run uvicorn main:app --port 10000", language="bash")

    st.divider()

    # -- File upload
    st.subheader("Upload Study Material")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}…"):
                try:
                    result = upload_pdf(uploaded_file.read(), uploaded_file.name)
                    st.session_state.upload_status = ("ok", result.get("message", "Upload successful"))
                except APIError as e:
                    st.session_state.upload_status = ("err", e.detail)
            st.rerun()

    # Show last upload result
    if st.session_state.upload_status:
        level, msg = st.session_state.upload_status
        if level == "ok":
            st.success(msg, icon="✅")
        else:
            st.error(msg, icon="❌")

    st.divider()

    # -- Documents in index
    st.subheader("Documents in Index")
    try:
        docs_resp = get_documents()
        docs = docs_resp.get("documents", [])
        if docs:
            for doc in docs:
                st.write(f"📄 {doc}")
        else:
            st.caption("No documents loaded yet.")
    except APIError:
        st.caption("Could not load document list.")

    st.divider()

    # -- Session controls
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("Clear Conversation", use_container_width=True):
        # We clear locally — server conversation also gets cleared via API
        try:
            from frontend.api_client import clear_conversation
            clear_conversation(st.session_state.session_id)
        except APIError:
            pass  # if server is down, still clear the local display
        st.session_state.messages = []
        st.session_state.upload_status = None
        st.rerun()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_quiz, tab_agent, tab_metrics = st.tabs(
    ["💬 Chat", "📝 Quiz", "🤖 Study Planner", "📊 Metrics"]
)

with tab_chat:
    chat_tab.render()

with tab_quiz:
    quiz_tab.render()

with tab_agent:
    agent_tab.render()

with tab_metrics:
    metrics_tab.render()
