"""
FastAPI endpoint tests.

Uses TestClient without a context manager so the lifespan (which tries to
load FAISS from disk) never runs. State is injected directly through the
autouse reset_state fixture in conftest.py + per-test setup via healthy_state.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from main import app

# Single shared client — no lifespan, no startup side effects
client = TestClient(app, raise_server_exceptions=True)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_healthy_when_vector_db_and_llm_ready(self, healthy_state):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["api_key_valid"] is True
        assert body["chunks_count"] == 100  # from make_mock_vector_db

    def test_degraded_when_no_vector_db(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"

    def test_degraded_when_no_llm(self, healthy_state):
        import app.state as state
        state.llm = None
        resp = client.get("/health")
        assert resp.json()["status"] == "degraded"
        assert resp.json()["api_key_valid"] is False

    def test_returns_corpus_chunk_count(self, healthy_state):
        import app.state as state
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["corpus_chunks"] == len(state.corpus_chunks)

    def test_retrieval_mode_hybrid_when_corpus_loaded(self, healthy_state):
        resp = client.get("/health")
        assert resp.json()["retrieval_mode"] == "hybrid_rerank"

    def test_retrieval_mode_vector_only_when_no_corpus(self, healthy_state):
        import app.state as state
        state.corpus_chunks = []
        resp = client.get("/health")
        assert resp.json()["retrieval_mode"] == "vector_only"


# ── /documents ────────────────────────────────────────────────────────────────

class TestDocumentsEndpoint:
    def test_returns_empty_list_when_no_docs(self):
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert resp.json()["documents"] == []

    def test_returns_loaded_documents(self, healthy_state):
        import app.state as state
        resp = client.get("/documents")
        assert resp.status_code == 200
        expected = {d.metadata["source"] for d in state.current_documents}
        assert set(resp.json()["documents"]) == expected


# ── /chat ─────────────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def test_503_when_no_documents_loaded(self):
        resp = client.post("/chat", json={"question": "What is attention?", "session_id": "s1"})
        assert resp.status_code == 503

    def test_503_when_llm_unavailable(self, healthy_state):
        import app.state as state
        state.llm = None
        state.qa_chain = None
        resp = client.post("/chat", json={"question": "test", "session_id": "s1"})
        assert resp.status_code == 503

    def test_returns_answer_and_sources(self, healthy_state):
        resp = client.post(
            "/chat",
            json={"question": "What is the attention mechanism?", "session_id": "chat_test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert isinstance(body["sources"], list)

    def test_session_id_echoed_in_response(self, healthy_state):
        resp = client.post("/chat", json={"question": "test", "session_id": "my_session"})
        assert resp.json()["session_id"] == "my_session"

    def test_default_session_used_when_none_provided(self, healthy_state):
        resp = client.post("/chat", json={"question": "test"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "default"


# ── /conversation ─────────────────────────────────────────────────────────────

class TestConversationEndpoint:
    def test_get_empty_history(self):
        resp = client.get("/conversation", params={"session_id": "new_session"})
        assert resp.status_code == 200
        assert resp.json()["history"] == []
        assert resp.json()["count"] == 0

    def test_delete_clears_history(self, healthy_state):
        import app.state as state
        state.session_histories["del_test"] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        resp = client.delete("/conversation", params={"session_id": "del_test"})
        assert resp.status_code == 200
        assert state.session_histories["del_test"] == []


# ── /quiz ─────────────────────────────────────────────────────────────────────

class TestQuizEndpoint:
    def test_generate_503_when_no_documents(self):
        resp = client.post("/quiz/generate", params={"session_id": "q1"})
        assert resp.status_code == 503

    def test_answer_400_when_no_active_quiz(self, healthy_state):
        resp = client.post(
            "/quiz/answer",
            json={"user_answer": "some answer", "session_id": "no_quiz_session"},
        )
        assert resp.status_code == 400

    def test_quiz_history_empty_for_new_session(self):
        resp = client.get("/quiz/history", params={"session_id": "fresh"})
        assert resp.status_code == 200
        assert resp.json()["history"] == []
        assert resp.json()["count"] == 0

    def test_quiz_history_returns_stored_items(self, healthy_state):
        import app.state as state
        state.quiz_history_store["hist_test"] = [
            {"question": "Q1", "answer": "A1", "timestamp": "2026-01-01T00:00:00"}
        ]
        resp = client.get("/quiz/history", params={"session_id": "hist_test"})
        assert resp.json()["count"] == 1
        assert resp.json()["history"][0]["question"] == "Q1"


# ── /agent/study-plan ─────────────────────────────────────────────────────────

class TestAgentEndpoint:
    def test_503_when_no_documents(self):
        resp = client.post(
            "/agent/study-plan",
            json={"goal": "Help me study attention", "session_id": "a1"},
        )
        assert resp.status_code == 503

    def test_503_when_no_llm(self, healthy_state):
        import app.state as state
        state.llm = None
        resp = client.post(
            "/agent/study-plan",
            json={"goal": "Help me study attention", "session_id": "a1"},
        )
        assert resp.status_code == 503
