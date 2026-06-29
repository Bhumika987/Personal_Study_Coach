import streamlit as st

from frontend.api_client import APIError, create_study_plan

_TOOL_ICONS = {
    "search_docs":           "🔍",
    "summarize_topic":       "📝",
    "generate_quiz_question": "❓",
}

_TOOL_LABELS = {
    "search_docs":           "Search Documents",
    "summarize_topic":       "Summarise Topic",
    "generate_quiz_question": "Generate Quiz Question",
}


def render() -> None:
    st.header("Study Planner Agent")
    st.caption(
        "Describe what you want to study. The agent autonomously searches your documents, "
        "summarises key concepts, and generates quiz questions — then writes a personalised study plan."
    )

    # ── Goal input ────────────────────────────────────────────────────────────
    goal = st.text_area(
        "What do you want to study today?",
        key="agent_goal",
        placeholder=(
            "e.g. Help me understand the attention mechanism and multi-head attention,\n"
            "      including how positional encoding works."
        ),
        height=110,
    )

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        run_clicked = st.button(
            "Generate Study Plan",
            type="primary",
            use_container_width=True,
            disabled=not (goal or "").strip(),
        )
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.study_plan = None
            st.rerun()

    if run_clicked and goal.strip():
        st.session_state.study_plan = None
        with st.spinner("Agent is thinking — searching documents, summarising, generating questions…"):
            try:
                result = create_study_plan(goal.strip(), st.session_state.session_id)
                st.session_state.study_plan = result
            except APIError as e:
                st.error(f"Error {e.status_code}: {e.detail}", icon="❌")
                return

    # ── Results ───────────────────────────────────────────────────────────────
    plan_data = st.session_state.get("study_plan")
    if not plan_data:
        return

    steps = plan_data.get("steps", [])
    plan_text = plan_data.get("plan", "")

    # Agent reasoning trace
    if steps:
        st.divider()
        st.subheader(f"Agent Reasoning — {len(steps)} tool call(s)")
        st.caption("Expand each step to see what the agent searched for and found.")

        for i, step in enumerate(steps, 1):
            tool_name = step["tool"]
            icon = _TOOL_ICONS.get(tool_name, "🔧")
            label = _TOOL_LABELS.get(tool_name, tool_name)
            header = f"{icon} Step {i}: {label}  —  `{step['input'][:60]}`"

            with st.expander(header):
                st.markdown(f"**Input:** `{step['input']}`")
                st.divider()
                output = step.get("output", "")
                # Truncate very long tool outputs in the UI — full text in plan
                display = output if len(output) <= 1200 else output[:1200] + "\n\n…_(truncated for display)_"
                st.markdown(display)

    # Final study plan
    st.divider()
    st.subheader("Your Personalised Study Plan")

    if plan_text:
        st.markdown(plan_text)
    else:
        st.warning("The agent did not produce a final plan. Try rephrasing your goal.")

    # Download button
    if plan_text:
        st.download_button(
            label="Download Study Plan (.md)",
            data=plan_text,
            file_name="study_plan.md",
            mime="text/markdown",
            use_container_width=True,
        )
