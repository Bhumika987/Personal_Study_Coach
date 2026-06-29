"""
Unit tests for the core retrieval functions.

bm25_search and reciprocal_rank_fusion are pure functions — no external
services, no mocking needed.
"""

import pytest
from langchain_core.documents import Document

from app.rag.retriever import bm25_search, reciprocal_rank_fusion


def doc(text: str) -> Document:
    return Document(page_content=text, metadata={})


# ── BM25 search ───────────────────────────────────────────────────────────────

class TestBM25Search:
    def test_returns_most_relevant_doc_first(self):
        corpus = [
            doc("attention mechanism maps queries to keys and values"),
            doc("python functions are defined with def keyword"),
            doc("gradient descent optimises neural network weights"),
        ]
        results = bm25_search("attention queries keys", corpus, k=3)
        assert results[0].page_content == corpus[0].page_content

    def test_empty_corpus_returns_empty_list(self):
        assert bm25_search("anything", [], k=5) == []

    def test_excludes_zero_score_docs(self):
        corpus = [
            doc("completely unrelated passage about cooking"),
            doc("another irrelevant paragraph about weather"),
        ]
        results = bm25_search("attention transformer self-attention", corpus, k=5)
        # Both docs should score 0 — none returned
        assert results == []

    def test_respects_k_limit(self):
        corpus = [doc(f"attention term {i} neural") for i in range(10)]
        results = bm25_search("attention neural", corpus, k=3)
        assert len(results) <= 3

    def test_case_insensitive(self):
        # Need 3+ docs so BM25 IDF for the matching term stays positive
        corpus = [
            doc("Attention Is All You Need paper by Vaswani"),
            doc("recurrent neural networks process sequences step by step"),
            doc("convolutional filters detect local patterns in images"),
        ]
        results_lower = bm25_search("attention", corpus, k=3)
        results_upper = bm25_search("ATTENTION", corpus, k=3)
        assert len(results_lower) > 0
        assert len(results_upper) > 0
        assert results_lower[0].page_content == results_upper[0].page_content


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def test_deduplicates_shared_documents(self):
        shared = doc("shared document appears in both lists")
        list1 = [shared, doc("only in list 1")]
        list2 = [doc("only in list 2"), shared]
        fused = reciprocal_rank_fusion([list1, list2])
        contents = [d.page_content for d in fused]
        assert contents.count(shared.page_content) == 1

    def test_top_ranked_in_both_lists_wins(self):
        winner = doc("top ranked in both lists")
        other = doc("lower ranked document")
        fused = reciprocal_rank_fusion([[winner, other], [winner, other]])
        assert fused[0].page_content == winner.page_content

    def test_single_list_returns_same_order(self):
        docs = [doc(f"doc {i}") for i in range(5)]
        fused = reciprocal_rank_fusion([docs])
        assert [d.page_content for d in fused] == [d.page_content for d in docs]

    def test_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_doc_in_more_lists_ranks_higher(self):
        boosted = doc("appears in three lists")
        competitor = doc("appears in one list only")
        result = reciprocal_rank_fusion([
            [boosted],
            [competitor, boosted],
            [boosted],
        ])
        assert result[0].page_content == boosted.page_content

    def test_rrf_k_constant_dampens_rank_effect(self):
        # With a large k, rank differences matter less
        d1 = doc("rank 1 doc")
        d2 = doc("rank 2 doc")
        fused_low_k = reciprocal_rank_fusion([[d1, d2]], k=1)
        fused_high_k = reciprocal_rank_fusion([[d1, d2]], k=1000)
        # Order should remain the same regardless of k
        assert fused_low_k[0].page_content == fused_high_k[0].page_content
