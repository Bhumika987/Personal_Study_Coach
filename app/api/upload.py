import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

import app.state as state
from app.config import MAX_UPLOAD_BYTES, UPLOAD_DIR
from app.rag.pipeline import process_documents

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed",
        )

    # Prevent path traversal — take only the bare filename, strip any directory components
    safe_name = os.path.basename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")

    file_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        # Stream in 64 KB chunks and enforce the size cap before writing to disk
        total_bytes = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                buffer.write(chunk)

        success = process_documents([file_path])
        if not success:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract content from the PDF",
            )

        return JSONResponse(content={
            "message": f"Successfully uploaded and processed {safe_name}",
            "filename": safe_name,
            "llm_available": state.llm is not None,
            "status": "success",
        })

    except HTTPException:
        raise  # Re-raise HTTP errors without wrapping them
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the uploaded file",
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@router.get("/documents")
async def list_documents():
    if not state.current_documents:
        return JSONResponse(content={"documents": [], "count": 0})
    doc_names = sorted(
        set(doc.metadata.get("source", "Unknown") for doc in state.current_documents)
    )
    return JSONResponse(content={"documents": doc_names, "count": len(doc_names)})


@router.delete("/index")
async def clear_index():
    """Wipe the in-memory state and all persisted index files so the user can
    start fresh with a new set of documents."""
    import shutil

    state.vector_db = None
    state.retriever = None
    state.qa_chain = None
    state.corpus_chunks = []
    state.current_documents = []
    state.session_histories = {}
    state.quiz_store = {}
    state.quiz_history_store = {}

    # Remove persisted FAISS + corpus files
    from app.config import FAISS_INDEX_DIR
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    logger.info("Index cleared by user request")
    return JSONResponse(content={"message": "Index cleared. Upload new PDFs to start fresh."})
