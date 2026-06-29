"""
Shared fixtures for all test modules.

State isolation: app.state is a mutable module-level singleton shared
across requests. The reset_state fixture runs before every test to ensure
no test bleeds into another.
"""

import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_doc(text: str, source: str = "test.pdf", page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def make_mock_llm(answer: str = "Mock answer.") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value.content = answer
    llm.bind_tools.return_value = llm  # so build_graph() works without error
    return llm


def make_mock_vector_db(ntotal: int = 100) -> MagicMock:
    vdb = MagicMock()
    vdb.index.ntotal = ntotal
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [
        make_doc("Dense retrieval result", "paper.pdf", 2)
    ]
    vdb.as_retriever.return_value = retriever_mock
    return vdb


def make_mock_retriever(docs=None) -> MagicMock:
    r = MagicMock()
    r.invoke.return_value = docs or [make_doc("Retrieved chunk about attention.")]
    return r


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Wipe shared state before (and after) every test."""
    import app.state as state

    state.vector_db = None
    state.retriever = None
    state.llm = None
    state.qa_chain = None
    state.corpus_chunks = []
    state.current_documents = []
    state.session_histories = {}
    state.quiz_store = {}
    state.quiz_history_store = {}

    yield

    # Teardown — same reset so the next test doesn't inherit anything
    state.vector_db = None
    state.retriever = None
    state.llm = None
    state.qa_chain = None
    state.corpus_chunks = []
    state.current_documents = []
    state.session_histories = {}
    state.quiz_store = {}
    state.quiz_history_store = {}


@pytest.fixture
def fake_docs():
    return [
        make_doc("The attention mechanism maps queries to keys and values.", "attention.pdf", 1),
        make_doc("Python functions are defined with the def keyword.", "python.pdf", 5),
        make_doc("Sequence-to-sequence models encode and decode sequences.", "seq2seq.pdf", 3),
        make_doc("Multi-head attention runs scaled dot-product attention in parallel.", "attention.pdf", 2),
    ]


@pytest.fixture
def healthy_state(fake_docs):
    """Populate app.state with mock objects so endpoints return 200."""
    import app.state as state

    state.vector_db = make_mock_vector_db()
    state.llm = make_mock_llm()
    state.retriever = make_mock_retriever(fake_docs[:2])
    state.qa_chain = MagicMock()
    state.qa_chain.invoke.return_value = "The attention mechanism is..."
    state.corpus_chunks = fake_docs
    # current_documents holds Document objects (the /documents endpoint reads .metadata)
    state.current_documents = [
        make_doc("chunk", "attention.pdf"),
        make_doc("chunk", "python.pdf"),
    ]
