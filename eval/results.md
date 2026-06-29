# Evaluation Results — Personal Study Coach

## System Configuration

| Component | Value |
|-----------|-------|
| LLM | llama-3.1-8b-instant (Groq) |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Vector store | FAISS (MMR, fetch_k=20, k=10) |
| BM25 | BM25Okapi via rank-bm25 |
| Retrieval strategy | Hybrid BM25 + FAISS → RRF → cross-encoder rerank |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Chunk size | 1000 chars / 200 overlap |
| Final chunks to LLM | 4 |
| Evaluator | RAGAS 0.4.x |

---

## RAGAS Scores

> Run `uv run python eval/run_eval.py` to generate real scores.
> Full per-question results are saved to `eval/ragas_scores.json`.

| Metric | Score | What it measures |
|--------|-------|-----------------|
| **Faithfulness** | — | Every claim in the answer is supported by retrieved context |
| **Answer Relevancy** | — | The answer directly addresses the question asked |
| **Context Precision** | — | Retrieved chunks are relevant to the question |
| **Context Recall** | — | Retrieved context contains all info needed to answer |

---

## Test Questions (10)

| # | Question | Source document |
|---|----------|----------------|
| 1 | What is the attention mechanism in neural networks? | attention1.pdf |
| 2 | How does self-attention work in the Transformer? | attention1.pdf |
| 3 | What are the advantages of self-attention over recurrent layers? | attention1.pdf |
| 4 | What is multi-head attention and why is it used? | attention1.pdf |
| 5 | What is positional encoding and why does the Transformer need it? | attention1.pdf |
| 6 | What is the difference between encoder and decoder in the Transformer? | attention1.pdf |
| 7 | What are sequence-to-sequence models used for? | sequence1.pdf |
| 8 | How do you define a function in Python? | PYTHON1.pdf |
| 9 | What are Python lists and how do you create one? | PYTHON1.pdf |
| 10 | What are large language models and how are they trained? | ll1.pdf |

---

## Retrieval Pipeline — Why Each Stage Exists

| Stage | Problem it solves |
|-------|------------------|
| **Query rewriting** | Vocabulary mismatch — user phrasing ≠ document phrasing |
| **BM25** | Exact keyword hits that dense embeddings compress away |
| **FAISS MMR** | Semantic similarity + diversity across retrieved chunks |
| **RRF fusion** | Merges two incompatible score scales using rank positions only |
| **Cross-encoder rerank** | High-precision relevance scoring on the small fused candidate set |

---

## Known Limitations

1. **Single-hop retrieval only** — questions requiring synthesis across 3+ chunks may produce incomplete answers
2. **BM25 rebuilt per query** — acceptable for ≤500 chunks; should be cached for larger corpora
3. **Session state is in-memory** — lost on server restart (Phase 5 will add SQLite persistence)
4. **No streaming** — LLM response is fully buffered before returning

---

## How to Run Evaluation

```bash
# Make sure the FAISS index exists first (start server once):
uv run uvicorn main:app

# Then run evaluation (takes ~3–5 minutes for 10 questions × 4 metrics):
uv run python eval/run_eval.py
```
