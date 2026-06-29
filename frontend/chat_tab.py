import streamlit as st

from frontend.api_client import APIError, chat


def render() -> None:
    st.header("Chat with your documents")
    st.caption(
        "Ask anything about the PDFs you've uploaded. "
        "Answers are grounded in the document — sources shown below each reply."
    )

    # ── Conversation display ──────────────────────────────────────────────────
    if not st.session_state.messages:
        st.info("Upload a PDF in the sidebar, then ask a question below.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"Sources — {len(sources)} chunk(s) retrieved"):
                    for src in sources:
                        st.caption(f"📄 **{src['file']}**  ·  Page {src['page']}")

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your study material..."):
        # Display user bubble immediately (before API call)
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = chat(prompt, st.session_state.session_id)
                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander(f"Sources — {len(sources)} chunk(s) retrieved"):
                            for src in sources:
                                st.caption(f"📄 **{src['file']}**  ·  Page {src['page']}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                except APIError as e:
                    st.error(f"Error {e.status_code}: {e.detail}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ {e.detail}"}
                    )
