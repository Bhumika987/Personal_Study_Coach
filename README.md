## 📘 Personal Study Coach API – RAG-Based Study Assistant

#### 🚀 Project Overview

The **Personal Study Coach API** is a FastAPI-based backend that enables users to upload study materials and interact with them using AI.

It supports:
- 📂 PDF upload and processing
- 💬 Context-aware question answering
- 🧠 Session-based conversation memory
- 📝 Quiz generation
- 📊 Answer evaluation with feedback

The system uses **Retrieval-Augmented Generation (RAG)** to ensure responses are strictly based on uploaded documents.

---

#### 🏗️ Architecture

```mermaid
flowchart TD
    A[Upload PDF] --> B[Document Loader]
    B --> C[Chunking]
    C --> D[Embeddings]
    D --> E[FAISS Vector DB]

    E --> F[Retriever (MMR)]
    F --> G[LLM (Llama 3.1 - Groq)]
    G --> H[Answer]

    subgraph Session
        I[Conversation History]
        J[Quiz Store]
        K[Quiz History]
    end

    I --> F
    G --> J
    G --> K

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
```
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

2. **Retrieval Accuracy**

   * Some queries returned less relevant chunks
   * Due to limited `k` and embedding constraints

3. **No Memory Handling**

   * System doesn’t retain conversation context
  
4. **Swagger UI Bug**
   * Multiple file upload not working properly, only working for single upload.
---
#### 👩‍💻 Author

**Bhumika Sahu**: Project is still in progress...
