import streamlit as st

from frontend.api_client import APIError, answer_quiz, generate_quiz, get_quiz_history


def _score_color(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 50:
        return "orange"
    return "red"


def render() -> None:
    st.header("Quiz yourself")
    st.caption("Questions are generated from your uploaded documents. Answers are evaluated semantically — partial credit is given.")

    col_main, col_history = st.columns([2, 1], gap="large")

    # ── Main quiz area ────────────────────────────────────────────────────────
    with col_main:
        if st.button("Generate New Question", type="primary", use_container_width=True):
            st.session_state.quiz = None
            st.session_state.quiz_result = None
            st.session_state.quiz_answer_input = ""
            with st.spinner("Selecting context and generating question..."):
                try:
                    result = generate_quiz(st.session_state.session_id)
                    st.session_state.quiz = result
                except APIError as e:
                    st.error(f"Error {e.status_code}: {e.detail}")

        if st.session_state.quiz:
            quiz = st.session_state.quiz

            # Question card
            st.subheader("Question")
            st.info(quiz["question"], icon="❓")

            if st.session_state.quiz_result is None:
                # Answer input phase
                user_answer = st.text_area(
                    "Your answer",
                    key="quiz_answer_input",
                    placeholder="Type your answer here...",
                    height=120,
                )

                hint_col, submit_col = st.columns([1, 1])
                with hint_col:
                    with st.expander("Need a hint?"):
                        st.caption(quiz.get("answer_hint", "No hint available."))
                with submit_col:
                    if st.button("Submit Answer", type="primary", use_container_width=True):
                        if not user_answer.strip():
                            st.warning("Please write an answer before submitting.")
                        else:
                            with st.spinner("Evaluating your answer..."):
                                try:
                                    result = answer_quiz(
                                        user_answer, st.session_state.session_id
                                    )
                                    st.session_state.quiz_result = result
                                    st.rerun()
                                except APIError as e:
                                    st.error(f"Error {e.status_code}: {e.detail}")

            else:
                # Result phase
                result = st.session_state.quiz_result
                score = result.get("score", 0)
                correct = result.get("correct", False)

                # Score card
                color = _score_color(score)
                if correct:
                    st.success(f"Correct!  Score: **{score} / 100**", icon="✅")
                else:
                    st.error(f"Not quite.  Score: **{score} / 100**", icon="❌")

                st.progress(score / 100)

                st.markdown(f"**Feedback:** {result.get('feedback', '')}")

                with st.expander("Show correct answer"):
                    st.write(result.get("expected_answer", ""))

                st.divider()
                if st.button("Try Another Question", use_container_width=True):
                    st.session_state.quiz = None
                    st.session_state.quiz_result = None
                    st.rerun()

        elif "quiz" not in st.session_state or st.session_state.quiz is None:
            st.markdown(
                "_Click **Generate New Question** to get started. "
                "Make sure a PDF is loaded first._"
            )

    # ── History sidebar ───────────────────────────────────────────────────────
    with col_history:
        st.subheader("Session History")
        try:
            history_data = get_quiz_history(st.session_state.session_id)
            items = history_data.get("history", [])
            if items:
                st.caption(f"{len(items)} question(s) this session")
                for item in reversed(items):
                    with st.expander(item["question"][:55] + "…"):
                        st.caption(f"🕐 {item['timestamp'][:19]}")
                        st.write(f"**Answer:** {item['answer'][:200]}")
            else:
                st.caption("No questions yet this session.")
        except APIError:
            st.caption("Could not load history.")
