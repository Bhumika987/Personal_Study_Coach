"""
Advanced retrieval pipeline — Phase 2.

Query flow:
  user question
    → query rewriting      (LLM makes it retrieval-friendly)
    → BM25 search          (keyword / sparse retrieval)
    → FAISS MMR search     (semantic / dense retrieval)
    → RRF fusion           (merge ranked lists without score calibration)
    → cross-encoder rerank (precision pass over fused candidates)
    → top-k chunks to LLM
"""

import logging
from typing import List, Optional

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import CANDIDATE_K, FINAL_TOP_K, RERANKER_MODEL, RRF_K

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ── Core retrieval functions ───────────────────────────────────────────────────

def rewrite_query(query: str, llm) -> str:
    """
    LLM rewrites the user's question to be keyword-rich and retrieval-optimized.
    Falls back to the original query on any failure so retrieval is never blocked.
    """
    try:
        prompt = (
            "Rewrite the following question into a concise, keyword-rich search query "
            "for retrieving relevant passages from a document. "
            "Output ONLY the rewritten query — no explanation, no quotes.\n\n"
            f"Original: {query}\nRewritten:"
        )
        rewritten = llm.invoke(prompt).content.strip()
        if rewritten:
            logger.info(f"Query rewrite: '{query}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
    return query


def bm25_search(query: str, corpus: List[Document], k: int = CANDIDATE_K) -> List[Document]:
    """
    Sparse keyword retrieval using BM25Okapi.
    BM25 complements dense embeddings by catching exact term matches that
    semantic search can miss (e.g. acronyms, proper nouns, formulas).
    """
    if not corpus:
        return []
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [doc.page_content.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())

        top_indices = np.argsort(scores)[::-1][:k]
        return [corpus[i] for i in top_indices if scores[i] > 0]
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        return []


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], k: int = RRF_K
) -> List[Document]:
    """
    Merges multiple ranked lists into one using Reciprocal Rank Fusion.

    RRF score formula:  score(d) = Σ  1 / (k + rank(d))
                                  lists

    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    It dampens the impact of very high ranks without needing score normalization,
    making it safe to combine BM25 scores and cosine distances directly.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in ranked_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content[:200]  # dedup by content prefix
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def rerank_docs(
    query: str, docs: List[Document], top_k: int = FINAL_TOP_K
) -> List[Document]:
    """
    Cross-encoder precision pass over fused candidates.

    Bi-encoders (used in FAISS) embed query and doc independently — fast but
    approximate. A cross-encoder jointly encodes the (query, doc) pair, producing
    a much more accurate relevance score. We use it only on the small fused
    candidate set (≤20 docs) to keep latency low.
    """
    if not docs:
        return docs
    try:
        reranker = get_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except Exception as e:
        logger.warning(f"Reranking failed, returning original order: {e}")
        return docs[:top_k]


# ── LangChain Retriever wrapper ───────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Drop-in LangChain retriever that runs the full advanced pipeline.
    Reads from app.state at query time so it always sees the latest
    vector store and corpus without needing to be recreated.
    """

    candidate_k: int = CANDIDATE_K
    final_k: int = FINAL_TOP_K
    rrf_k: int = RRF_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        import app.state as state

        if not state.vector_db:
            return []

        # Step 1 — rewrite for better retrieval (skip if no LLM)
        search_query = rewrite_query(query, state.llm) if state.llm else query

        # Step 2 — BM25 keyword search
        bm25_results = bm25_search(search_query, state.corpus_chunks, k=self.candidate_k)

        # Step 3 — FAISS dense search
        vector_results = state.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.candidate_k,
                "fetch_k": self.candidate_k * 2,
                "lambda_mult": 0.7,
            },
        ).invoke(search_query)

        # Step 4 — RRF fusion
        fused = reciprocal_rank_fusion([vector_results, bm25_results], k=self.rrf_k)

        # Step 5 — cross-encoder rerank
        final = rerank_docs(query, fused, top_k=self.final_k)

        logger.info(
            f"Retrieval | BM25: {len(bm25_results)} "
            f"vector: {len(vector_results)} "
            f"fused: {len(fused)} "
            f"final: {len(final)}"
        )
        return final

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
