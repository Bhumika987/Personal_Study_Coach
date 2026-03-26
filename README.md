# 📚 Personal Study Coach (RAG-based AI Assistant)
#### 🚀 Overview
This project is a **Retrieval-Augmented Generation (RAG) based Study Assistant** that helps users learn from their study material.
It allows users to:
* 📖 Upload study material (PDF/TXT)
* ❓ Ask questions and get contextual answers
* 🧠 Generate quiz questions
* ✅ Evaluate answers with feedback
* 💬 Interact in a simple study flow (explain → quiz → evaluate)

---

### 🏗️ Architecture

```text
User → FastAPI → Retriever → FAISS Vector DB → LLM (Groq) → Response
```

#### Flow Explanation:

1. User uploads or queries study material
2. Documents are chunked into smaller parts
3. Chunks are converted into embeddings
4. Stored in FAISS vector database
5. Relevant chunks retrieved using MMR
6. LLM generates answer using retrieved context

---

### 🔍 RAG Pipeline

##### 1. Document Loading

* PDFs loaded using `PyPDFLoader`
* Metadata (source, page) added for citation

##### 2. Chunking

* Chunk size: **1000 characters**
* Overlap: **200 characters**
* Strategy: RecursiveCharacterTextSplitter

##### 3. Embeddings

* Model: `sentence-transformers/all-MiniLM-L6-v2`
* Dimension: 384

##### 4. Vector Database

* FAISS used for fast similarity search

##### 5. Retrieval

* Method: **MMR (Maximum Marginal Relevance)**
* Ensures:

  * Relevant chunks
  * Diversity in results

##### 6. LLM

* Model: Groq (LLaMA 3.1)
* Used for:

  * Answer generation
  * Quiz creation
  * Answer evaluation

---
### ⚙️ Features

##### ✅ 1. Question Answering
* Answers based ONLY on document context
* Includes source citations

##### 📘 2. Explanation Mode
* Explains topics in detail from study material

##### 🧠 3. Quiz Generation
* Generates questions from document
* Avoids repetition (basic logic)

##### 📊 4. Answer Evaluation
* Checks correctness
* Provides feedback + score
---

### 🌐 API Endpoints

| Method | Endpoint | Description      |
| ------ | -------- | ---------------- |
| POST   | /upload  | Upload PDF       |
| POST   | /chat    | Ask question     |
| GET    | /health  | Check API status |

---

## 📊 Evaluation
```bash
eval/results.md
```
Includes:

* Question
* Answer
* Sources
* Comments

---

#### ⚠️ Challenges Faced

1. **API Key Exposure**
   * Accidentally committed `.env`
   * GitHub blocked push due to secret detection
   * Solved by:

     * Removing `.env`
     * Cleaning notebook outputs
     * Rewriting git history

2. **Jupyter Notebook Issue**

   * API key remained in hidden outputs
   * Fixed using:

     * Restart & Clear Output
     * `nbstripout`

3. **Retrieval Accuracy**

   * Some queries returned less relevant chunks
   * Due to limited `k` and embedding constraints

4. **No Memory Handling**

   * System doesn’t retain conversation context
  
5. **Swagger UI Bug**
   * Multiple file upload not working properly, only working for single upload.
---
#### 👩‍💻 Author

**Bhumika Sahu**: Project is still in progress...
