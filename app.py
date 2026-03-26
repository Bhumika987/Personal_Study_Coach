import os
import shutil
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, status
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
conversation_history = []  # Simple conversation memory

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    conversation_id: str

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_count: int
    api_key_valid: bool
    timestamp: str

class QuizResponse(BaseModel):
    question: str
    answer_hint: str  # For demo purposes

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
    """Process multiple PDF documents and create vector store"""
    global vector_db, retriever, qa_chain, current_documents, llm
    
    documents = []
    
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
            
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {paper_name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not documents:
        logger.error("No documents loaded")
        return False
    
    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Create embeddings and vector store
    embedding_model = get_embedding_model()
    vector_db = FAISS.from_documents(chunks, embedding_model)
    
    # Save the index
    vector_db.save_local(FAISS_INDEX_DIR)
    
    # Create retriever
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )
    
    # Initialize LLM and create QA chain
    llm = initialize_llm()
    if llm:
        qa_chain = create_qa_chain(retriever, llm)
    else:
        qa_chain = None
    
    current_documents = documents
    
    return True

@app.on_event("startup")
async def startup_event():
    """Load default documents on startup"""
    # First check if API key is available
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("⚠️  No GROQ_API_KEY found in .env file!")
        logger.warning("Please create .env file with: GROQ_API_KEY=your_key_here")
        logger.warning("You can get a key from: https://console.groq.com/keys")
    
    default_docs = ["data/attention1.pdf", "data/sequence1.pdf", "data/PYTHON1.pdf"]
    existing_docs = [doc for doc in default_docs if os.path.exists(doc)]
    
    if existing_docs:
        logger.info(f"Loading default documents: {existing_docs}")
        success = process_documents(existing_docs)
        if success:
            logger.info("✅ Default documents loaded successfully")
        else:
            logger.error("❌ Failed to load default documents")
    else:
        logger.warning("No default documents found")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and RAG system is ready"""
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
    global llm, qa_chain
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the uploaded file
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
        # Clean up temp file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about the study material"""
    global conversation_history
    
    if not vector_db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents loaded. Please upload a PDF first."
        )
    
    if not qa_chain or not llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not initialized. Please check your GROQ_API_KEY in .env file"
        )
    
    try:
        # Add conversation context if we have history
        if conversation_history and request.conversation_id:
            context_prompt = f"\nPrevious conversation:\n{conversation_history[-3:]}\n"
            question = context_prompt + request.question
        else:
            question = request.question
        
        # Get answer from RAG system
        answer = qa_chain.invoke(question)
        
        # Get sources
        docs = retriever.invoke(request.question)
        sources = []
        for doc in docs[:3]:  # Limit to top 3 sources
            sources.append({
                "file": doc.metadata.get("source", "Unknown"),
                "page": str(doc.metadata.get("page", "N/A"))
            })
        
        # Update conversation history
        conversation_history.append({
            "question": request.question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 conversations
        if len(conversation_history) > 10:
            conversation_history.pop(0)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=request.conversation_id or "default"
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        if "API key" in str(e) or "401" in str(e):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key. Please check your GROQ_API_KEY in .env file"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )

@app.post("/quiz/generate", response_model=QuizResponse)
async def generate_quiz():
    """Generate a quiz question from the study material"""
    if not vector_db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents loaded. Please upload a PDF first."
        )
    
    if not llm:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not initialized. Please check your GROQ_API_KEY in .env file"
        )
    
    try:
        # Get random chunks for context
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
        
        # Parse response
        lines = response.content.strip().split('\n')
        question = ""
        answer = ""
        
        for line in lines:
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
        
        return QuizResponse(
            question=question,
            answer_hint=answer[:100] + "..." if len(answer) > 100 else answer
        )
        
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        if "API key" in str(e) or "401" in str(e):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key. Please check your GROQ_API_KEY in .env file"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating quiz: {str(e)}"
        )

# Add the rest of your endpoints (explain, quiz/check, etc.) here
# ... [keep the same as before]

@app.get("/documents")
async def list_documents():
    """List currently loaded documents"""
    if not current_documents:
        return JSONResponse(
            content={
                "documents": [],
                "count": 0
            }
        )
    
    # Get unique document names
    doc_names = set()
    for doc in current_documents:
        doc_names.add(doc.metadata.get("source", "Unknown"))
    
    return JSONResponse(
        content={
            "documents": list(doc_names),
            "count": len(doc_names)
        }
    )

@app.get("/conversation")
async def get_conversation_history():
    """Get conversation history"""
    return JSONResponse(
        content={
            "history": conversation_history,
            "count": len(conversation_history)
        }
    )

@app.delete("/conversation")
async def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return JSONResponse(
        content={
            "message": "Conversation history cleared",
            "status": "success"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)