from typing import Any, Dict, List

# RAG pipeline objects
vector_db: Any = None
retriever: Any = None
llm: Any = None
qa_chain: Any = None
current_documents: List = []

# Per-session conversation history  {session_id: [{"role": ..., "content": ...}]}
session_histories: Dict[str, List[Dict]] = {}

# Active quiz per session           {session_id: {"question": ..., "answer": ...}}
quiz_store: Dict[str, Dict] = {}

# Full quiz history per session     {session_id: [{"question": ..., "answer": ..., "timestamp": ...}]}
quiz_history_store: Dict[str, List[Dict]] = {}
