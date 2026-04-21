"""
api.py — FastAPI backend exposing the RAG agent as a REST API.

Endpoints:
    POST /query          Run a question through the full LangGraph agent
    GET  /health         Health check

Run with:
    uvicorn api:app --reload
"""

import time
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import rag_agent

load_dotenv()

app = FastAPI(
    title="EU AI Act Compliance Agent",
    description=(
        "A LangGraph-powered RAG agent for querying the EU AI Act. "
        "Uses adaptive retrieval, document grading, and hallucination detection."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What are the obligations for high-risk AI systems?"}
            ]
        }
    }


class QueryResponse(BaseModel):
    question: str
    answer: str
    steps: List[str]
    web_search_used: bool
    latency_seconds: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "EU AI Act RAG Agent"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Run the user's question through the LangGraph RAG agent.

    The agent will:
    1. Route the question to vector store or web search
    2. Retrieve and grade relevant documents
    3. Generate an answer
    4. Verify the answer is grounded and resolves the question
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    start = time.perf_counter()

    initial_state = {
        "question": request.question,
        "generation": "",
        "web_search": "No",
        "documents": [],
        "steps": [],
        "retries": 0,
    }

    try:
        final_state = rag_agent.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency = round(time.perf_counter() - start, 2)

    return QueryResponse(
        question=request.question,
        answer=final_state.get("generation", "No answer generated."),
        steps=final_state.get("steps", []),
        web_search_used=final_state.get("web_search") == "Yes",
        latency_seconds=latency,
    )
