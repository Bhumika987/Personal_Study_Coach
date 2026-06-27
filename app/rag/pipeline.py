import json
import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import app.state as state
from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
)

logger = logging.getLogger(__name__)

_CORPUS_PATH = os.path.join(FAISS_INDEX_DIR, "corpus.json")

# Cached singleton — avoids reloading the 80 MB model on every upload
_embedding_model: HuggingFaceEmbeddings | None = None


# ── Model helpers ─────────────────────────────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def initialize_llm() -> ChatGroq | None:
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found in environment")
        return None
    try:
        llm = ChatGroq(
            temperature=0.3,
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            max_retries=2,
        )
        llm.invoke("Say 'OK'")
        logger.info("LLM initialized and tested successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None


def create_qa_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Answer questions based ONLY on the provided context.

Context:
{context}

Question: {input}

Instructions:
1. Answer ONLY using information from the context
2. If the answer isn't in the context, say "I don't have enough information to answer this"
3. Include citations using (Source: [filename], Page [page])
4. Be concise but comprehensive

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "input": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )


# ── Corpus persistence (for BM25 survival across restarts) ────────────────────

def save_corpus(chunks: List[Document]) -> None:
    """Serialise chunk text + metadata to JSON so BM25 can be rebuilt after restart."""
    try:
        data = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
        with open(_CORPUS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"Corpus saved: {len(chunks)} chunks → {_CORPUS_PATH}")
    except Exception as e:
        logger.warning(f"Failed to save corpus: {e}")


def load_corpus() -> List[Document]:
    """Load persisted chunk corpus for BM25 indexing on startup."""
    if not os.path.exists(_CORPUS_PATH):
        return []
    try:
        with open(_CORPUS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        chunks = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]
        logger.info(f"Corpus loaded: {len(chunks)} chunks from {_CORPUS_PATH}")
        return chunks
    except Exception as e:
        logger.warning(f"Failed to load corpus: {e}")
        return []


# ── Document processing ───────────────────────────────────────────────────────

def process_documents(file_paths: List[str]) -> bool:
    from app.rag.retriever import HybridRetriever

    all_new_docs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            paper_name = os.path.basename(file_path)
            for doc in docs:
                doc.metadata["source"] = paper_name
                doc.metadata["page"] = doc.metadata.get("page", 0)
            all_new_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {paper_name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    if not all_new_docs:
        logger.error("No documents could be loaded from the provided paths")
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    new_chunks = splitter.split_documents(all_new_docs)
    logger.info(f"Created {len(new_chunks)} chunks")

    # ── FAISS (dense index) ──
    embedding_model = get_embedding_model()
    if state.vector_db is None:
        state.vector_db = FAISS.from_documents(new_chunks, embedding_model)
        logger.info("Created new FAISS vector store")
    else:
        state.vector_db.add_documents(new_chunks)
        logger.info(f"Added {len(new_chunks)} chunks to existing FAISS store")
    state.vector_db.save_local(FAISS_INDEX_DIR)

    # ── BM25 corpus (sparse index) ──
    state.corpus_chunks.extend(new_chunks)
    save_corpus(state.corpus_chunks)

    # ── Wire retriever + QA chain ──
    if state.llm is None:
        state.llm = initialize_llm()

    state.retriever = HybridRetriever()
    state.qa_chain = create_qa_chain(state.retriever, state.llm) if state.llm else None

    state.current_documents.extend(all_new_docs)
    return True
