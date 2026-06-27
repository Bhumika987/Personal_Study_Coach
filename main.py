import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import DEFAULT_DOCS, FAISS_INDEX_DIR, GROQ_API_KEY
from app.rag.pipeline import (
    create_qa_chain,
    get_embedding_model,
    initialize_llm,
    process_documents,
)
import app.state as state
from app.api.chat import router as chat_router
from app.api.quiz import router as quiz_router
from app.api.upload import router as upload_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY found — LLM features will be unavailable")

    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if os.path.exists(index_file):
        try:
            from langchain_community.vectorstores import FAISS

            embedding_model = get_embedding_model()
            state.vector_db = FAISS.load_local(
                FAISS_INDEX_DIR,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"Loaded FAISS index — {state.vector_db.index.ntotal} vectors")

            state.retriever = state.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7},
            )
            state.llm = initialize_llm()
            if state.llm:
                state.qa_chain = create_qa_chain(state.retriever, state.llm)
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
    else:
        existing = [d for d in DEFAULT_DOCS if os.path.exists(d)]
        if existing:
            logger.info(f"Loading default documents: {existing}")
            process_documents(existing)
        else:
            logger.warning("No default documents found. Upload a PDF to get started.")

    yield  # Server runs here


app = FastAPI(
    title="Personal Study Coach API",
    description="RAG-based study assistant for PDF documents",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(quiz_router)


class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_count: int
    api_key_valid: bool
    timestamp: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    chunks_count = state.vector_db.index.ntotal if state.vector_db else 0
    return HealthResponse(
        status="healthy" if state.vector_db and state.llm else "degraded",
        documents_loaded=len(state.current_documents),
        chunks_count=chunks_count,
        api_key_valid=state.llm is not None,
        timestamp=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
