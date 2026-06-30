import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

UPLOAD_DIR = "temp_uploads"
FAISS_INDEX_DIR = "faiss_index"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# Advanced retrieval settings
CANDIDATE_K = 10          # results pulled from each retriever (BM25 + vector) before fusion
FINAL_TOP_K = 4           # chunks passed to the LLM after reranking
RRF_K = 60                # constant in Reciprocal Rank Fusion formula
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_DOCS = ["data/attention1.pdf", "data/sequence1.pdf", "data/PYTHON1.pdf", "data/ll1.pdf"]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
