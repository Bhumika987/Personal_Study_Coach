import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import app.state as state
from app.agent.graph import run_study_planner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent")


class StudyPlanRequest(BaseModel):
    goal: str
    session_id: Optional[str] = None


class StudyStep(BaseModel):
    tool: str
    input: str
    output: str


class StudyPlanResponse(BaseModel):
    plan: str
    steps: List[StudyStep]
    tool_count: int
    session_id: str


@router.post("/study-plan", response_model=StudyPlanResponse)
async def create_study_plan(req: StudyPlanRequest):
    if not state.vector_db:
        raise HTTPException(503, "No documents loaded. Upload a PDF first.")
    if not state.llm:
        raise HTTPException(503, "LLM not available. Check your GROQ_API_KEY.")

    sid = req.session_id or "default"

    try:
        result = run_study_planner(req.goal, state.llm)
        return StudyPlanResponse(
            plan=result["plan"],
            steps=[StudyStep(**s) for s in result["steps"]],
            tool_count=result["tool_count"],
            session_id=sid,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Study planner agent failed: {e}")
        raise HTTPException(500, "Study planner agent encountered an error. Please try again.")
