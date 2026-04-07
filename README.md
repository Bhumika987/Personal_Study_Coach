Perfect — this format is clean and structured 👍
Here’s your **final polished, copy-paste ready README** (fixed + aligned with your actual code + no Mermaid errors):

---

````markdown
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

E --> F[Retriever - MMR]
F --> G[LLM - Llama 3.1 Groq]
G --> H[Answer]

subgraph Session
    I[Conversation History]
    J[Quiz Store]
    K[Quiz History]
end

I --> F
G --> J
G --> K
````

---

### 🔍 RAG Pipeline

##### 1. Document Loading

* PDFs loaded using `PyPDFLoader`
* Metadata (source, page) added for citation

##### 2. Chunking

* Chunk size: **1000 characters**
* Overlap: **200 characters**
* Strategy: `RecursiveCharacterTextSplitter`

##### 3. Embeddings

* Model: `sentence-transformers/all-MiniLM-L6-v2`
* Dimension: **384**

##### 4. Vector Database

* FAISS used for fast similarity search
* Supports **incremental updates (no overwrite)**

##### 5. Retrieval

* Method: **MMR (Maximum Marginal Relevance)**
* Ensures:

  * Relevant chunks
  * Diversity in results

##### 6. LLM

* Model: Groq (**llama-3.1-8b-instant**)
* Used for:

  * Answer generation
  * Quiz creation
  * Answer evaluation

---

### ⚙️ Features

##### ✅ 1. Question Answering

* Answers based ONLY on document context
* Includes source citations (file + page)
* Uses session-based context (last interactions)

##### 🧠 2. Conversation Memory

* Maintains chat history per `session_id`
* Stores last 20 messages
* Enables contextual follow-up questions

##### 📝 3. Quiz Generation

* Generates questions from document chunks
* Stores correct answer in backend
* Returns hint to user
* Tracks quiz history per session

##### 📊 4. Answer Evaluation

* Semantic evaluation using LLM
* Returns:

  * Correct / Incorrect
  * Feedback
  * Score (0–100)

---

### 🌐 API Endpoints

| Method | Endpoint       | Description             |
| ------ | -------------- | ----------------------- |
| POST   | /upload        | Upload PDF              |
| POST   | /chat          | Ask question            |
| POST   | /quiz/generate | Generate quiz           |
| POST   | /quiz/answer   | Submit quiz answer      |
| GET    | /quiz/history  | Get quiz history        |
| GET    | /documents     | List uploaded documents |
| GET    | /conversation  | Get chat history        |
| DELETE | /conversation  | Clear chat history      |
| GET    | /health        | Check API status        |

---

## 📊 Evaluation

```bash
eval/results.md
```

Includes:

* Questions
* Expected vs actual answers
* Correctness evaluation
* Comments

---

#### ⚠️ Challenges Faced

1. **API Key Exposure**

   * Accidentally committed `.env`
   * GitHub blocked push due to secret detection
   * Solved by:

     * Removing `.env`
     * Cleaning outputs
     * Rewriting git history

2. **Retrieval Accuracy**

   * Some queries returned irrelevant chunks
   * Due to embedding limitations and fixed `k` value

3. **Session Management Complexity**

   * Initially used global memory (caused leakage)
   * Fixed using `session_id`-based storage

4. **Quiz Handling**

   * Managing current quiz + history per session required separate stores

5. **Swagger UI Limitation**

   * File upload works best for single file at a time

---

#### 👩‍💻 Author

**Bhumika Sahu**
GenAI Associate Assessment
Project in progress...

👉 or make your README look **🔥 premium with badges + visuals**
```
