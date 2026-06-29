"""
LangGraph ReAct study planner.

build_graph() is called at request time with the live LLM instance so the
graph always reflects the current state — no stale captures.

The agent follows a ReAct loop:
  [agent] --tool_calls?--> [tools] --> [agent] --> ... --> final answer
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.agent.tools import STUDY_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Personal Study Coach — an intelligent agent that creates personalised study plans from uploaded documents.

When the student gives you a study goal, follow this exact sequence:
1. Call search_docs first to discover what content is available about the topic.
2. Call summarize_topic for each major concept the student needs to understand (usually 2-3 calls).
3. Call generate_quiz_question for 2-3 key concepts so the student can test themselves.
4. After all tool calls are complete, write a structured Study Plan with these sections:

   ## Overview
   Brief description of the topic and what the student will learn.

   ## Key Concepts
   Bullet-point summaries of each concept (drawn from your summarize_topic calls).

   ## Practice Questions
   The quiz questions you generated, numbered.

   ## Suggested Study Sequence
   A numbered list telling the student what to study in what order.

Rules:
- Only use information returned by the tools — never fabricate content.
- Always make at least one search_docs call and at least one summarize_topic call.
- Keep summaries concise and student-friendly.
- If a tool returns "No content found", note that in the plan and move on."""


def build_graph(llm: Any):
    """Create a compiled LangGraph ReAct graph bound to the given LLM instance."""
    return create_react_agent(
        model=llm,
        tools=STUDY_TOOLS,
        state_modifier=SystemMessage(content=SYSTEM_PROMPT),
    )


def _extract_steps(messages: list) -> list[dict]:
    """Pull tool call → tool result pairs out of the message history."""
    steps = []
    # Map tool_call_id → step index so we can fill in output when ToolMessage arrives
    id_to_idx: dict[str, int] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.get("args", {})
                # Both tools use either 'query' or 'topic' as their single arg
                input_str = args.get("query") or args.get("topic") or str(args)
                idx = len(steps)
                steps.append(
                    {"tool": tc["name"], "input": input_str, "output": ""}
                )
                id_to_idx[tc["id"]] = idx

        elif isinstance(msg, ToolMessage):
            idx = id_to_idx.get(msg.tool_call_id)
            if idx is not None:
                steps[idx]["output"] = msg.content

    return steps


def _final_answer(messages: list) -> str:
    """Return the last AIMessage content that is not a tool call."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            return msg.content
    return ""


def run_study_planner(goal: str, llm: Any) -> dict:
    """
    Run the ReAct agent for a study goal.

    Returns:
        plan       — the agent's final markdown study plan
        steps      — list of {tool, input, output} dicts
        tool_count — number of tool invocations
    """
    graph = build_graph(llm)
    try:
        result = graph.invoke({"messages": [HumanMessage(content=goal)]})
    except Exception as e:
        logger.error(f"LangGraph agent error: {e}")
        raise

    messages = result.get("messages", [])
    steps = _extract_steps(messages)
    plan = _final_answer(messages)

    logger.info(
        f"Study planner finished: {len(steps)} tool calls, "
        f"{len(plan)} chars in plan"
    )
    return {"plan": plan, "steps": steps, "tool_count": len(steps)}
