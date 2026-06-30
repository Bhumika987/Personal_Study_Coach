"""
RAGAS evaluation for the Personal Study Coach RAG pipeline.

Metrics:
  Faithfulness       — Is every claim in the answer supported by the retrieved context?
  Answer Relevancy   — Does the answer actually address the question asked?
  Context Precision  — Are the retrieved chunks relevant to the question?
  Context Recall     — Does the retrieved context contain everything needed to answer?

Usage:
  uv run python eval/run_eval.py

Output:
  - Prints a score table to stdout
  - Writes eval/ragas_scores.json  (full results per question)
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# ── bootstrap path so we can import app.* from the project root ──────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

# ── project imports ───────────────────────────────────────────────────────────
from app.config import FAISS_INDEX_DIR, GROQ_API_KEY
import app.state as state
from app.rag.pipeline import (
    create_qa_chain,
    get_embedding_model,
    initialize_llm,
    load_corpus,
)
from app.rag.retriever import HybridRetriever
from eval.test_set import TEST_QUESTIONS


# ── RAG system loader ─────────────────────────────────────────────────────────

def load_rag_system() -> None:
    """Load vector store, corpus, LLM, and retriever from disk."""
    from langchain_community.vectorstores import FAISS

    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if not os.path.exists(index_file):
        print(
            f"\nERROR: No FAISS index found at '{index_file}'.\n"
            "Start the server once so it builds the index from default documents:\n"
            "  uv run uvicorn main:app\n"
        )
        sys.exit(1)

    print("Loading FAISS index ...", end=" ", flush=True)
    state.vector_db = FAISS.load_local(
        FAISS_INDEX_DIR, get_embedding_model(), allow_dangerous_deserialization=True
    )
    print(f"{state.vector_db.index.ntotal} vectors")

    print("Loading BM25 corpus  ...", end=" ", flush=True)
    state.corpus_chunks = load_corpus()
    print(f"{len(state.corpus_chunks)} chunks")

    print("Initializing LLM     ...", end=" ", flush=True)
    state.llm = initialize_llm()
    if not state.llm:
        print("\nERROR: LLM initialization failed. Check your GROQ_API_KEY in .env")
        sys.exit(1)
    print("OK")

    state.retriever = HybridRetriever()
    state.qa_chain = create_qa_chain(state.retriever, state.llm)
    print("RAG system ready.\n")


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_question(question: str) -> tuple[str, list[str]]:
    """Return (answer, list_of_context_strings) for one question."""
    answer = state.qa_chain.invoke(question)
    docs = state.retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    return answer, contexts


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def build_ragas_dataset(rows: list[dict]):
    """Convert pipeline outputs to a RAGAS EvaluationDataset."""
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)


def run_ragas(dataset) -> dict:
    """Run all four RAGAS metrics using our Groq LLM + local embeddings.

    RAGAS 0.4 'collections' metrics require InstructorLLM (not LangchainLLMWrapper).
    We build one via llm_factory pointed at Groq's OpenAI-compatible endpoint.
    Embeddings use RAGAS's native HuggingFaceEmbeddings so no wrapper is needed.
    """
    from openai import OpenAI
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    ragas_llm = llm_factory("llama-3.1-8b-instant", provider="openai", client=groq_client)
    ragas_emb = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
    )
    return dict(result)


# ── Output helpers ────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "faithfulness":      "Faithfulness      (answer grounded in context?)",
    "answer_relevancy":  "Answer Relevancy  (answers the question?)",
    "context_precision": "Context Precision (retrieved chunks relevant?)",
    "context_recall":    "Context Recall    (context covers the answer?)",
}


def print_score_table(scores: dict) -> None:
    width = 52
    print("\n" + "═" * width)
    print("  RAGAS EVALUATION  —  Personal Study Coach")
    print("  Retrieval mode : hybrid BM25 + FAISS + rerank")
    print("  LLM evaluator  : llama-3.1-8b-instant (Groq)")
    print("═" * width)
    for key, label in METRIC_LABELS.items():
        val = scores.get(key)
        if val is None:
            print(f"  {label}: N/A")
            continue
        bar_len = int(float(val) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {label}")
        print(f"    {bar}  {float(val):.3f}")
    print("═" * width + "\n")


def save_results(rows: list[dict], scores: dict) -> Path:
    out = {
        "timestamp": datetime.now().isoformat(),
        "retrieval_mode": "hybrid_rerank",
        "llm": "llama-3.1-8b-instant",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "num_questions": len(rows),
        "scores": {k: round(float(v), 4) for k, v in scores.items() if v is not None},
        "per_question": [
            {
                "question": r["question"],
                "answer": r["answer"][:300] + "..." if len(r["answer"]) > 300 else r["answer"],
                "num_contexts": len(r["contexts"]),
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ],
    }
    path = PROJECT_ROOT / "eval" / "ragas_scores.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    load_rag_system()

    # Step 1 — run each question through the RAG pipeline
    n = len(TEST_QUESTIONS)
    print(f"Running {n} questions through the RAG pipeline ...\n")
    rows = []
    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["question"]
        print(f"  [{i:2d}/{n}] {q[:65]}{'...' if len(q) > 65 else ''}")
        answer, contexts = run_question(q)
        rows.append({
            "question":     q,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })

    # Step 2 — evaluate with RAGAS
    print("\nRunning RAGAS evaluation (this makes LLM calls for each metric) ...\n")
    dataset = build_ragas_dataset(rows)
    scores = run_ragas(dataset)

    # Step 3 — display + persist
    print_score_table(scores)
    out_path = save_results(rows, scores)
    print(f"Full results saved → {out_path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
