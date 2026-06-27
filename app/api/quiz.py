import logging
import random
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import app.state as state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quiz")


class QuizResponse(BaseModel):
    question: str
    answer_hint: str


class QuizAnswerRequest(BaseModel):
    user_answer: str
    session_id: Optional[str] = None


class QuizAnswerResponse(BaseModel):
    correct: bool
    feedback: str
    score: int
    expected_answer: str


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(session_id: Optional[str] = Query(None)):
    if not state.vector_db:
        raise HTTPException(503, "No documents loaded. Please upload a PDF first.")
    if not state.llm:
        raise HTTPException(503, "LLM not available. Check your GROQ_API_KEY.")

    sid = session_id or "default"

    try:
        # Use a meaningful seed query instead of an empty string to avoid biased sampling
        seed_queries = [
            "key concepts and definitions",
            "important principles and theory",
            "examples and applications",
            "processes and mechanisms",
        ]
        all_docs = state.vector_db.similarity_search(random.choice(seed_queries), k=20)
        context = random.choice(all_docs).page_content

        quiz_prompt = f"""Based on this study material, generate a quiz question:

Context: {context[:1500]}

Format your response EXACTLY as:
QUESTION: [Your question here]
ANSWER: [The correct answer here]"""

        response = state.llm.invoke(quiz_prompt)
        lines = response.content.strip().split("\n")

        question = ""
        answer = ""
        for line in lines:
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()

        if not question or not answer:
            raise HTTPException(500, "LLM did not return a valid quiz format. Please try again.")

        state.quiz_store[sid] = {"question": question, "answer": answer}

        if sid not in state.quiz_history_store:
            state.quiz_history_store[sid] = []
        state.quiz_history_store[sid].append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        })

        hint = answer[:100] + "..." if len(answer) > 100 else answer
        return QuizResponse(question=question, answer_hint=hint)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        raise HTTPException(500, "Failed to generate quiz question")


@router.post("/answer", response_model=QuizAnswerResponse)
async def answer_quiz(req: QuizAnswerRequest):
    sid = req.session_id or "default"

    if sid not in state.quiz_store:
        raise HTTPException(400, "No active quiz for this session. Generate a quiz first.")

    # Guard added: llm can be None if API key was missing at startup
    if not state.llm:
        raise HTTPException(503, "LLM not available. Check your GROQ_API_KEY.")

    quiz = state.quiz_store[sid]
    correct_answer = quiz["answer"]

    eval_prompt = f"""Question: {quiz['question']}
Correct answer: {correct_answer}
User's answer: {req.user_answer}

Evaluate if the user's answer is correct (consider semantic similarity, not exact match).
Respond with exactly three lines:
CORRECT: yes/no
FEEDBACK: (brief feedback)
SCORE: (0-100 integer)"""

    try:
        response = state.llm.invoke(eval_prompt)
        lines = response.content.strip().split("\n")

        correct = False
        feedback = ""
        score = 0
        for line in lines:
            if line.startswith("CORRECT:"):
                correct = "yes" in line.lower()
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()
            elif line.startswith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:", "").strip())
                except ValueError:
                    score = 0

        return QuizAnswerResponse(
            correct=correct,
            feedback=feedback,
            score=score,
            expected_answer=correct_answer,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating quiz answer: {e}")
        raise HTTPException(500, "Failed to evaluate answer")


@router.get("/history")
async def get_quiz_history(session_id: Optional[str] = Query(None)):
    sid = session_id or "default"
    history = state.quiz_history_store.get(sid, [])
    return {"session_id": sid, "count": len(history), "history": history}
