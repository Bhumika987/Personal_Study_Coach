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

DEFAULT_DOCS = ["data/attention1.pdf", "data/sequence1.pdf", "data/PYTHON1.pdf"]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
