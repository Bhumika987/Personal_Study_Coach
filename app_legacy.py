import os
import shutil
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# Check API key immediately
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not found in environment variables!")
    print("Please create a .env file with: GROQ_API_KEY=your_key_here")
else:
    print(f"✅ GROQ_API_KEY loaded (starts with: {GROQ_API_KEY[:10]}...)")

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Personal Study Coach API",
    description="RAG-based study assistant for PDF documents",
    version="1.0.0"
)

# Global variables to store the RAG system state
vector_db = None
retriever = None
llm = None
qa_chain = None
current_documents = []
#conversation_history = []  # Simple conversation memory (global, not per session)

# Per‑session conversation history
session_histories = {}  # session_id -> list of dicts
# Store quizzes per session (so users can answer later)
quiz_store = {}  # session_id -> {"question": str, "answer": str}
# Store full quiz history per session (list of dicts)
quiz_history_store = {}  # session_id -> [{"question": ..., "answer": ..., "timestamp": ...}]

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    session_id: str

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_count: int
    api_key_valid: bool
    timestamp: str

class QuizResponse(BaseModel):
    question: str
    answer_hint: str  # Short hint, full answer is stored in backend

class QuizAnswerRequest(BaseModel):
    user_answer: str
    session_id: Optional[str] = None

class QuizAnswerResponse(BaseModel):
    correct: bool
    feedback: str
    score: int
    expected_answer: str  # Optionally show the correct answer after evaluation

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
UPLOAD_DIR = "temp_uploads"
FAISS_INDEX_DIR = "faiss_index"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

def initialize_llm():
    """Initialize the LLM with error handling"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found in environment")
        return None
    
    try:
        llm = ChatGroq(
            temperature=0.3,
            model=GROQ_MODEL,
            api_key=api_key,
            max_retries=2
        )
        # Test the connection
        test_response = llm.invoke("Say 'OK'")
        if test_response:
            logger.info("✅ LLM initialized and tested successfully")
            return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None
    
    return None

def get_embedding_model():
    """Get the embedding model"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def create_qa_chain(retriever, llm):
    """Create the QA chain"""
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
    
    qa_chain = (
        {
            "context": retriever | format_docs,
            "input": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

def process_documents(file_paths: List[str]):
    """Process multiple PDF documents and add to vector store (incremental)."""
    global vector_db, retriever, qa_chain, current_documents, llm
    
    all_new_docs = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add metadata
            paper_name = os.path.basename(file_path)
            for doc in docs:
                doc.metadata["source"] = paper_name
                doc.metadata["page"] = doc.metadata.get("page", 0)
            
            all_new_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {paper_name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not all_new_docs:
        logger.error("No documents loaded")
        return False
    
    # Chunk the new documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    new_chunks = text_splitter.split_documents(all_new_docs)
    logger.info(f"Created {len(new_chunks)} new chunks")
    
    embedding_model = get_embedding_model()
    
    # If vector DB already exists, add to it; otherwise create new
    if vector_db is None:
        vector_db = FAISS.from_documents(new_chunks, embedding_model)
        logger.info("Created new vector store")
    else:
        vector_db.add_documents(new_chunks)
        logger.info(f"Added {len(new_chunks)} chunks to existing vector store")
    
    # Save the updated index
    vector_db.save_local(FAISS_INDEX_DIR)
    
    # Update retriever
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
    )
    
    # Recreate QA chain if needed
    if llm is None:
        llm = initialize_llm()
    if llm:
        qa_chain = create_qa_chain(retriever, llm)
    else:
        qa_chain = None
    
    # Keep a list of all loaded documents (optional, for listing)
    current_documents.extend(all_new_docs)
    
    return True

@app.on_event("startup")
async def startup_event():
    """Load existing FAISS index and default documents if index is empty."""
    global vector_db, retriever, qa_chain, llm, current_documents
    
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("⚠️  No GROQ_API_KEY found in .env file!")
    
    # Try to load existing FAISS index
    index_path = FAISS_INDEX_DIR
    if os.path.exists(os.path.join(index_path, "index.faiss")):
        try:
            embedding_model = get_embedding_model()
            vector_db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            logger.info(f"✅ Loaded existing FAISS index with {vector_db.index.ntotal} vectors")
            
            # Recreate retriever and QA chain
            retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
            )
            llm = initialize_llm()
            if llm:
                qa_chain = create_qa_chain(retriever, llm)
            else:
                qa_chain = None
                
            # Also rebuild current_documents list (optional, for /documents endpoint)
            # This is tricky because we don't store full docs in the index. 
            # We can either skip or store metadata separately.
            # For simplicity, we'll just note that /documents might not show old docs.
            return
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
    
    # If no index or load failed, load default documents
    default_docs = ["data/attention1.pdf", "data/sequence1.pdf", "data/PYTHON1.pdf"]
    existing_docs = [doc for doc in default_docs if os.path.exists(doc)]
    if existing_docs:
        logger.info(f"Loading default documents: {existing_docs}")
        process_documents(existing_docs)
    else:
        logger.warning("No default documents found. Please upload a PDF.")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    chunks_count = 0
    if vector_db:
        chunks_count = vector_db.index.ntotal
    
    api_key_valid = llm is not None
    
    return HealthResponse(
        status="healthy" if vector_db and llm else "degraded",
        documents_loaded=len(current_documents),
        chunks_count=chunks_count,
        api_key_valid=api_key_valid,
        timestamp=datetime.now().isoformat()
    )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file to be used as study material"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        success = process_documents([file_path])
        
        if success:
            return JSONResponse(
                content={
                    "message": f"Successfully uploaded and processed {file.filename}",
                    "filename": file.filename,
                    "llm_available": llm is not None,
                    "status": "success"
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process the PDF"
            )
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not vector_db:
        raise HTTPException(503, "No documents loaded.")
    if not qa_chain or not llm:
        raise HTTPException(503, "LLM not initialized.")
    
    sid = request.session_id or "default"
    
    # Get or create history for this session
    if sid not in session_histories:
        session_histories[sid] = []
    history = session_histories[sid]
    
    try:
        # Add previous conversation context (last 3 exchanges) if any
        if history:
            last_msgs = history[-6:]  # last 3 Q&A pairs
            context_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_msgs])
            question = f"Previous conversation:\n{context_prompt}\n\nNew question: {request.question}"
        else:
            question = request.question
        
        answer = qa_chain.invoke(question)
        docs = retriever.invoke(request.question)
        sources = [{"file": d.metadata.get("source", "Unknown"), "page": str(d.metadata.get("page", "N/A"))} for d in docs[:3]]
        
        # Store in session history
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": answer})
        # Keep only last 20 messages (10 exchanges)
        if len(history) > 20:
            session_histories[sid] = history[-20:]
        
        return ChatResponse(answer=answer, sources=sources, session_id=sid)
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(500, str(e))
    

# ---------- UPDATED QUIZ ENDPOINTS ----------
@app.post("/quiz/generate", response_model=QuizResponse)
async def generate_quiz(session_id: Optional[str] = Query(None)):
    """Generate a quiz question for the given session (stores full answer)."""
    if not vector_db:
        raise HTTPException(503, "No documents loaded. Please upload a PDF first.")
    if not llm:
        raise HTTPException(503, "LLM not initialized. Check your GROQ_API_KEY.")
    
    sid = session_id or "default"
    
    try:
        import random
        all_docs = vector_db.similarity_search("", k=20)
        context = random.choice(all_docs).page_content
        
        quiz_prompt = f"""
        Based on this study material, generate a quiz question:
        
        Context: {context[:1500]}
        
        Format your response EXACTLY as:
        QUESTION: [Your question here]
        ANSWER: [The correct answer here]
        """
        
        response = llm.invoke(quiz_prompt)
        lines = response.content.strip().split('\n')
        question = ""
        answer = ""
        for line in lines:
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
        
        # Store current quiz for answering
        quiz_store[sid] = {"question": question, "answer": answer}
        
        # --- NEW: Store in history ---
        if sid not in quiz_history_store:
            quiz_history_store[sid] = []
        quiz_history_store[sid].append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Return only question + short hint (full answer stored server-side)
        hint = answer[:100] + "..." if len(answer) > 100 else answer
        return QuizResponse(question=question, answer_hint=hint)
        
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        if "API key" in str(e) or "401" in str(e):
            raise HTTPException(401, "Invalid or missing API key.")
        raise HTTPException(500, f"Error generating quiz: {str(e)}")
    
@app.post("/quiz/answer", response_model=QuizAnswerResponse)
async def answer_quiz(req: QuizAnswerRequest):
    """Submit an answer to the last generated quiz for the session."""
    sid = req.session_id or "default"
    
    if sid not in quiz_store:
        raise HTTPException(400, "No active quiz for this session. Please generate a quiz first.")
    
    quiz = quiz_store[sid]
    correct_answer = quiz["answer"]
    
    # Use LLM to evaluate semantically
    eval_prompt = f"""
    Question: {quiz['question']}
    Correct answer: {correct_answer}
    User's answer: {req.user_answer}
    
    Evaluate if the user's answer is correct (consider semantic similarity, not exact match).
    Respond with exactly three lines:
    CORRECT: yes/no
    FEEDBACK: (brief feedback)
    SCORE: (0-100 integer)
    """
    
    try:
        response = llm.invoke(eval_prompt)
        lines = response.content.strip().split('\n')
        correct = False
        feedback = ""
        score = 0
        
        for line in lines:
            if line.startswith("CORRECT:"):
                correct = "yes" in line.lower()
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()
            elif line.startswith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:", "").strip())
                except:
                    score = 0
        
        # Optionally keep the quiz in store for repeated attempts? Delete after use if desired.
        # For now, keep it; user can generate a new one.
        
        return QuizAnswerResponse(
            correct=correct,
            feedback=feedback,
            score=score,
            expected_answer=correct_answer  # Show correct answer after evaluation
        )
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        raise HTTPException(500, f"Error evaluating answer: {str(e)}")

@app.get("/quiz/history")
async def get_quiz_history(session_id: Optional[str] = Query(None)):
    """Get all previously generated quizzes for a session (including questions and answers)."""
    sid = session_id or "default"
    history = quiz_history_store.get(sid, [])
    return {
        "session_id": sid,
        "count": len(history),
        "history": history
    }
# ---------- Existing document & conversation endpoints ----------
@app.get("/documents")
async def list_documents():
    if not current_documents:
        return JSONResponse(content={"documents": [], "count": 0})
    doc_names = set(doc.metadata.get("source", "Unknown") for doc in current_documents)
    return JSONResponse(content={"documents": list(doc_names), "count": len(doc_names)})

@app.get("/conversation")
async def get_conversation_history(session_id: Optional[str] = Query(None)):
    sid = session_id or "default"
    history = session_histories.get(sid, [])
    return {"session_id": sid, "history": history, "count": len(history)}

@app.delete("/conversation")
async def clear_conversation(session_id: Optional[str] = Query(None)):
    sid = session_id or "default"
    if sid in session_histories:
        session_histories[sid] = []
    return {"message": f"Conversation history cleared for session {sid}", "status": "success"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)