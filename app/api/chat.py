import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import app.state as state

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    session_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not state.vector_db:
        raise HTTPException(503, "No documents loaded. Please upload a PDF first.")
    if not state.qa_chain or not state.llm:
        raise HTTPException(503, "LLM not available. Check your GROQ_API_KEY.")

    sid = request.session_id or "default"

    if sid not in state.session_histories:
        state.session_histories[sid] = []
    history = state.session_histories[sid]

    try:
        if history:
            # Include last 3 exchanges (6 messages) for context
            context_prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in history[-6:]
            )
            question = f"Previous conversation:\n{context_prompt}\n\nNew question: {request.question}"
        else:
            question = request.question

        answer = state.qa_chain.invoke(question)
        docs = state.retriever.invoke(request.question)
        sources = [
            {
                "file": d.metadata.get("source", "Unknown"),
                "page": str(d.metadata.get("page", "N/A")),
            }
            for d in docs[:3]
        ]

        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 20:
            state.session_histories[sid] = history[-20:]

        return ChatResponse(answer=answer, sources=sources, session_id=sid)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(500, "Failed to generate a response")


@router.get("/conversation")
async def get_conversation_history(session_id: Optional[str] = Query(None)):
    sid = session_id or "default"
    history = state.session_histories.get(sid, [])
    return {"session_id": sid, "history": history, "count": len(history)}


@router.delete("/conversation")
async def clear_conversation(session_id: Optional[str] = Query(None)):
    sid = session_id or "default"
    state.session_histories[sid] = []
    return {"message": f"Conversation history cleared for session {sid}", "status": "success"}
