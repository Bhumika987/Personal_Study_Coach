# 📘 Personal Study Coach API – RAG-Based Study Assistant
### 🚀 Project Overview

The **Personal Study Coach API** is a FastAPI-based backend that allows users to upload study materials (PDF/TXT) and interact with them using AI.  

It supports:

- 📂 Upload and incremental addition of multiple PDF/TXT files  
- 💬 Context-aware question answering with source citations  
- 🧠 Session-based conversation memory (each `session_id` has its own history)  
- 📝 Automatic quiz generation from the uploaded material  
- 📊 Answer evaluation with semantic feedback and score (0–100)  

The system uses **Retrieval-Augmented Generation (RAG)** to ensure responses are strictly based on the uploaded documents, avoiding hallucinations.

---

#### 🏗️ Architecture

![RAG-Based Study Assistant Architecture](./architecture.png)

---

#### 🔍 RAG Pipeline – Chunking Explanation

Chunking is critical for retrieval quality. I used `RecursiveCharacterTextSplitter` with:

| Parameter | Value | Reason |
|----------|------|--------|
| Chunk size | 1000 characters | Balances context richness and retrieval granularity |
| Overlap | 200 characters (20%) | Maintains continuity between chunks |
| Separators | `["\n\n", "\n", " ", ""]` | Preserves natural language structure |

---

#### ⚙️ Features

###### ✅ Question Answering
- Answers strictly from uploaded document context  
- Returns **sources (filename + page number)**  
- If answer not found → `"I don't have enough information"`

###### 🧠 Conversation Memory
- Per-session using `session_id`
- Stores last 20 messages
- Enables follow-up questions

###### 📝 Quiz Generation
- Generates quiz from document chunks
- Stores correct answer internally
- Returns hint to user
- Avoids repetition per session

###### 📊 Answer Evaluation
- Semantic evaluation using LLM
- Returns:
  - Correct / Incorrect  
  - Feedback  
  - Score (0–100)  
  - Expected answer  

###### 📂 Multiple PDF Support
- Incremental FAISS updates
- No overwriting of previous data
- Persistent storage

---

#### 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload PDF/TXT |
| POST | /chat | Ask question |
| POST | /quiz/generate | Generate quiz |
| POST | /quiz/answer | Submit answer |
| GET | /quiz/history | Quiz history |
| GET | /documents | List documents |
| GET | /conversation | Chat history |
| DELETE | /conversation | Clear history |
| GET | /health | API status |

---

#### 📊 Evaluation

Includes:
- 10 test questions  
- Expected vs actual answers  
- Correctness analysis  
- Comments  

---

#### ⚙️ Setup Instructions

```bash
# Clone repo
git clone https://github.com/your-username/study-coach.git
cd study-coach

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r req.txt

# Create .env file
echo GROQ_API_KEY=your_key_here > .env

# Run server
uvicorn app:app --reload
```

#### 🧠 Challenges Faced & Solutions

| Challenge | Solution |
|-----------|----------|
| **Loss of previous documents** after new upload | Implemented incremental FAISS update – `add_documents()` adds chunks without overwriting. |
| **Conversation history mixing across users** | Switched to per‑session storage using `session_id`. |
| **Hallucinations when answer not in context** | Engineered a strict prompt forcing `"I don't have enough information"`; added confidence scoring. |
| **Repeated quiz questions** | Maintained per‑session `quiz_history` and prompted the LLM to avoid last 5 questions. |
| **Slow retrieval with many chunks** | Used MMR with `fetch_k=15` and `k=4` to limit candidates while preserving diversity. |

---

#### 🔮 Future Improvements

- Hybrid search (BM25 + semantic) for better retrieval
- Cross‑encoder reranking to improve precision
- Persistent session storage (Redis / database) across server restarts
- Support for larger context windows (e.g., 2000‑character chunks)
- User authentication to isolate sessions more securely

---

#### 👤 Author

**Bhumika Sahu** – April 2026  
Project is still in progress...
