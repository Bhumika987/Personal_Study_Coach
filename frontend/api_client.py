"""
Centralised HTTP client for the Personal Study Coach FastAPI backend.
All tab modules import from here — nowhere else makes raw requests.

API_BASE_URL env var lets you point at Render/Railway in production
without changing any code.
"""

import os

import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:10000").rstrip("/")
TIMEOUT = 90  # LLM + reranker can take several seconds


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _handle(resp: requests.Response) -> dict:
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text or f"HTTP {resp.status_code}"
        raise APIError(resp.status_code, detail)
    return resp.json()


def _get(path: str, **kwargs) -> dict:
    try:
        return _handle(requests.get(f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs))
    except requests.exceptions.ConnectionError:
        raise APIError(0, f"Cannot reach API at {API_BASE} — is the server running?")
    except requests.exceptions.Timeout:
        raise APIError(408, "Request timed out. The server may be busy.")


def _post(path: str, **kwargs) -> dict:
    try:
        return _handle(requests.post(f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs))
    except requests.exceptions.ConnectionError:
        raise APIError(0, f"Cannot reach API at {API_BASE} — is the server running?")
    except requests.exceptions.Timeout:
        raise APIError(408, "Request timed out. The LLM may be busy.")


def _delete(path: str, **kwargs) -> dict:
    try:
        return _handle(requests.delete(f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs))
    except requests.exceptions.ConnectionError:
        raise APIError(0, f"Cannot reach API at {API_BASE} — is the server running?")


# ── Public API ────────────────────────────────────────────────────────────────

def health() -> dict:
    return _get("/health")


def get_documents() -> dict:
    return _get("/documents")


def upload_pdf(file_bytes: bytes, filename: str) -> dict:
    return _post(
        "/upload",
        files={"file": (filename, file_bytes, "application/pdf")},
    )


def chat(question: str, session_id: str) -> dict:
    return _post("/chat", json={"question": question, "session_id": session_id})


def clear_conversation(session_id: str) -> dict:
    return _delete("/conversation", params={"session_id": session_id})


def generate_quiz(session_id: str) -> dict:
    # POST with session_id as query param (no request body)
    return _post("/quiz/generate", params={"session_id": session_id})


def answer_quiz(user_answer: str, session_id: str) -> dict:
    return _post(
        "/quiz/answer",
        json={"user_answer": user_answer, "session_id": session_id},
    )


def get_quiz_history(session_id: str) -> dict:
    return _get("/quiz/history", params={"session_id": session_id})


def create_study_plan(goal: str, session_id: str) -> dict:
    return _post("/agent/study-plan", json={"goal": goal, "session_id": session_id})


def clear_index() -> dict:
    return _delete("/index")
