"""
api.py — FastAPI backend exposing the RAG agent as a REST API.

Endpoints:
    POST /query          Run a question through the full LangGraph agent
    GET  /health         Health check

Run with:
    uvicorn api:app --reload
"""

import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import Response

from agent.graph import rag_agent

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("eu_ai_act.api")


def _get_cors_origins() -> list[str]:
    raw_origins = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:8501,http://127.0.0.1:8501",
    )
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["http://localhost:8501"]


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
    allow_origins=_get_cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def add_request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed request_id=%s method=%s path=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            latency_ms,
        )
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%d latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What are the obligations for high-risk AI systems?"}]
        }
    }


class QueryResponse(BaseModel):
    question: str
    answer: str
    steps: list[str]
    web_search_used: bool
    latency_seconds: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "EU AI Act RAG Agent", "version": app.version}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request):
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
        request_id = getattr(http_request.state, "request_id", "unknown")
        logger.exception("agent_invoke_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later.",
        ) from exc

    latency = round(time.perf_counter() - start, 2)

    request_id = getattr(http_request.state, "request_id", "unknown")
    logger.info(
        "query_completed request_id=%s latency_s=%.2f web_search_used=%s question_chars=%d",
        request_id,
        latency,
        final_state.get("web_search") == "Yes",
        len(request.question),
    )

    return QueryResponse(
        question=request.question,
        answer=final_state.get("generation", "No answer generated."),
        steps=final_state.get("steps", []),
        web_search_used=final_state.get("web_search") == "Yes",
        latency_seconds=latency,
    )
