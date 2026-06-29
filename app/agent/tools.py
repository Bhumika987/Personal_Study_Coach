"""
LangChain tools given to the Study Planner agent.

Each function imports app.state at call time (not at module load) to avoid
circular imports and to always use the current live state.
"""

from langchain_core.tools import tool


@tool
def search_docs(query: str) -> str:
    """Search the document corpus for passages relevant to a topic or question.
    Use this first to understand what content is available before summarising."""
    import app.state as state

    if not state.retriever:
        return "No documents loaded. Please upload a PDF first."
    docs = state.retriever.invoke(query)
    if not docs:
        return f"No relevant passages found for '{query}'."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[Chunk {i} | {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@tool
def summarize_topic(topic: str) -> str:
    """Retrieve relevant passages from the documents and produce a clear,
    student-friendly summary of a topic. Call this for each major concept."""
    import app.state as state

    if not state.retriever or not state.llm:
        return "System not ready — documents or LLM unavailable."
    docs = state.retriever.invoke(topic)
    if not docs:
        return f"No content found about '{topic}' in the loaded documents."
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        f"Using ONLY the context below, write a clear and concise summary of '{topic}' "
        f"that would help a student understand the key ideas. "
        f"Use bullet points where appropriate.\n\n"
        f"Context:\n{context[:3000]}\n\n"
        f"Summary:"
    )
    response = state.llm.invoke(prompt)
    return response.content


@tool
def generate_quiz_question(topic: str) -> str:
    """Generate a quiz question and its model answer for a specific topic,
    based on content from the loaded study documents."""
    import app.state as state

    if not state.retriever or not state.llm:
        return "System not ready — documents or LLM unavailable."
    docs = state.retriever.invoke(topic)
    if not docs:
        return f"No content found about '{topic}' to base a quiz question on."
    context = docs[0].page_content[:1500]
    prompt = (
        f"Based on the study material below, create ONE clear quiz question about '{topic}' "
        f"with a thorough model answer.\n\n"
        f"Material:\n{context}\n\n"
        f"Respond EXACTLY in this format:\n"
        f"QUESTION: <the question>\n"
        f"ANSWER: <the model answer>"
    )
    response = state.llm.invoke(prompt)
    return response.content


# Exported list — graph.py and api/agent.py import this
STUDY_TOOLS = [search_docs, summarize_topic, generate_quiz_question]
