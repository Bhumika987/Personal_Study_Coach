"""
Unit tests for the Study Planner agent tools.

Tools read app.state at call time, so monkeypatching state attributes
is sufficient — no need to patch imports.
"""

import pytest
from langchain_core.documents import Document

from app.agent.tools import generate_quiz_question, search_docs, summarize_topic


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "paper.pdf", "page": 1})


# ── search_docs ───────────────────────────────────────────────────────────────

class TestSearchDocsTool:
    def test_returns_message_when_no_retriever(self, monkeypatch):
        import app.state as state
        monkeypatch.setattr(state, "retriever", None)
        result = search_docs.invoke({"query": "attention mechanism"})
        assert "No documents loaded" in result

    def test_returns_message_when_no_results(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        monkeypatch.setattr(state, "retriever", mock_retriever)
        result = search_docs.invoke({"query": "nonexistent topic xyz"})
        assert "No relevant passages found" in result

    def test_formats_chunks_with_source_and_page(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="attention content", metadata={"source": "paper.pdf", "page": 3})
        ]
        monkeypatch.setattr(state, "retriever", mock_retriever)
        result = search_docs.invoke({"query": "attention"})
        assert "attention content" in result
        assert "paper.pdf" in result
        assert "3" in result

    def test_numbers_multiple_chunks(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_doc(f"chunk {i}") for i in range(3)]
        monkeypatch.setattr(state, "retriever", mock_retriever)
        result = search_docs.invoke({"query": "test"})
        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "Chunk 3" in result


# ── summarize_topic ───────────────────────────────────────────────────────────

class TestSummarizeTopicTool:
    def test_returns_message_when_no_retriever(self, monkeypatch):
        import app.state as state
        monkeypatch.setattr(state, "retriever", None)
        monkeypatch.setattr(state, "llm", None)
        result = summarize_topic.invoke({"topic": "transformers"})
        assert "not ready" in result.lower() or "unavailable" in result.lower()

    def test_returns_message_when_no_content_found(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        monkeypatch.setattr(state, "retriever", mock_retriever)
        monkeypatch.setattr(state, "llm", MagicMock())
        result = summarize_topic.invoke({"topic": "unknown topic xyz"})
        assert "No content found" in result

    def test_returns_llm_summary(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_doc("Attention uses queries, keys, values.")]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Summary: attention is a mechanism."
        monkeypatch.setattr(state, "retriever", mock_retriever)
        monkeypatch.setattr(state, "llm", mock_llm)
        result = summarize_topic.invoke({"topic": "attention"})
        assert result == "Summary: attention is a mechanism."

    def test_passes_topic_in_llm_prompt(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_doc("some content")]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "ok"
        monkeypatch.setattr(state, "retriever", mock_retriever)
        monkeypatch.setattr(state, "llm", mock_llm)
        summarize_topic.invoke({"topic": "multi-head attention"})
        call_args = mock_llm.invoke.call_args[0][0]
        assert "multi-head attention" in call_args


# ── generate_quiz_question ────────────────────────────────────────────────────

class TestGenerateQuizQuestionTool:
    def test_returns_message_when_no_content(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        monkeypatch.setattr(state, "retriever", mock_retriever)
        monkeypatch.setattr(state, "llm", MagicMock())
        result = generate_quiz_question.invoke({"topic": "missing topic"})
        assert "No content found" in result

    def test_returns_llm_quiz_output(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_doc("Attention computes weighted sum over values.")]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "QUESTION: What does attention compute?\nANSWER: A weighted sum over values."
        )
        monkeypatch.setattr(state, "retriever", mock_retriever)
        monkeypatch.setattr(state, "llm", mock_llm)
        result = generate_quiz_question.invoke({"topic": "attention"})
        assert "QUESTION:" in result
        assert "ANSWER:" in result

    def test_uses_only_first_doc_for_context(self, monkeypatch):
        import app.state as state
        from unittest.mock import MagicMock
        docs = [_doc("first doc content"), _doc("second doc content")]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "QUESTION: test\nANSWER: test"
        monkeypatch.setattr(state, "retriever", mock_retriever)
        monkeypatch.setattr(state, "llm", mock_llm)
        generate_quiz_question.invoke({"topic": "attention"})
        prompt = mock_llm.invoke.call_args[0][0]
        assert "first doc content" in prompt
        assert "second doc content" not in prompt
