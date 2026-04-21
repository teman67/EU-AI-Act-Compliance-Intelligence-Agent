"""
nodes.py — Individual node functions for the EU AI Act RAG agent graph.

Each function receives the current GraphState and returns a dict of
state updates to merge back in.
"""

from typing import Any, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from agent.prompts import (
    ANSWER_GRADER_PROMPT,
    DOC_GRADER_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    RAG_PROMPT,
    ROUTER_PROMPT,
)
from agent.state import GraphState

MAX_RETRIES = 2
CHROMA_DIR = "./chroma_db"

# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------


class RouteQuery(BaseModel):
    """Router output — which data source to use."""

    datasource: str = Field(description="'vectorstore' or 'web_search'")


class GradeDocuments(BaseModel):
    """Relevance grade for a retrieved document."""

    binary_score: str = Field(description="'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Hallucination grade for a generated answer."""

    binary_score: str = Field(description="'yes' if grounded, 'no' if hallucinated")


class GradeAnswer(BaseModel):
    """Quality grade — does the answer resolve the question."""

    binary_score: str = Field(description="'yes' if resolves, 'no' if not")


# ---------------------------------------------------------------------------
# Shared LLM + tool instances (initialised lazily on first import)
# ---------------------------------------------------------------------------


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="eu_ai_act",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def _get_web_search_tool():
    return TavilySearchResults(max_results=3)


# ---------------------------------------------------------------------------
# Node: route_question
# ---------------------------------------------------------------------------


def route_question(state: GraphState) -> str:
    """
    Decides whether to route the question to the vector store or web search.
    Returns the name of the next node as a string (used as conditional edge).
    """
    llm = _get_llm().with_structured_output(RouteQuery)
    chain = ROUTER_PROMPT | llm
    result = cast(RouteQuery, chain.invoke({"question": state["question"]}))
    return result.datasource  # "vectorstore" or "web_search"


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------


def retrieve(state: GraphState) -> dict[str, Any]:
    """Retrieve relevant chunks from the ChromaDB vector store."""
    retriever = _get_retriever()
    docs = retriever.invoke(state["question"])
    return {
        "documents": [doc.page_content for doc in docs],
        "steps": ["🔍 Retrieved documents from vector store"],
    }


# ---------------------------------------------------------------------------
# Node: web_search
# ---------------------------------------------------------------------------


def web_search(state: GraphState) -> dict[str, Any]:
    """Run a Tavily web search and add results to documents."""
    search = _get_web_search_tool()
    results = search.invoke({"query": state["question"]})
    web_docs = [r["content"] for r in results]

    # Merge with any existing documents (e.g. from a previous retrieve step)
    existing = state.get("documents") or []
    return {
        "documents": existing + web_docs,
        "web_search": "Yes",
        "steps": ["🌐 Fetched results via web search"],
    }


# ---------------------------------------------------------------------------
# Node: grade_documents
# ---------------------------------------------------------------------------


def grade_documents(state: GraphState) -> dict[str, Any]:
    """
    Filter retrieved documents, keeping only those relevant to the question.
    Sets web_search="Yes" if too few relevant docs remain, triggering a fallback.
    """
    llm = _get_llm().with_structured_output(GradeDocuments)
    chain = DOC_GRADER_PROMPT | llm

    relevant_docs = []
    for doc in state["documents"]:
        result = cast(
            GradeDocuments,
            chain.invoke(
                {
                    "document": doc,
                    "question": state["question"],
                }
            ),
        )
        if result.binary_score == "yes":
            relevant_docs.append(doc)

    trigger_web = "Yes" if len(relevant_docs) < 2 else "No"
    step = f"📋 Graded {len(state['documents'])} documents → " f"{len(relevant_docs)} relevant" + (
        " → triggering web search fallback" if trigger_web == "Yes" else ""
    )
    return {
        "documents": relevant_docs,
        "web_search": trigger_web,
        "steps": [step],
    }


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------


def generate(state: GraphState) -> dict[str, Any]:
    """Generate an answer from the graded documents using the RAG prompt."""
    llm = _get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()

    context = "\n\n---\n\n".join(state["documents"])
    generation = chain.invoke(
        {
            "context": context,
            "question": state["question"],
        }
    )
    retries = state.get("retries", 0)
    return {
        "generation": generation,
        "retries": retries,
        "steps": [f"✍️  Generated answer (attempt {retries + 1})"],
    }


# ---------------------------------------------------------------------------
# Node: grade_generation (hallucination + answer quality check)
# ---------------------------------------------------------------------------


def grade_generation(state: GraphState) -> str:
    """
    Two-stage quality check:
      1. Is the generation grounded in documents? (hallucination check)
      2. Does it actually answer the question?

    Returns the name of the next node as a string for conditional routing.
    """
    llm = _get_llm()
    retries = state.get("retries", 0)

    # --- Hallucination check ---
    hall_grader = HALLUCINATION_GRADER_PROMPT | llm.with_structured_output(GradeHallucinations)
    hall_result = cast(
        GradeHallucinations,
        hall_grader.invoke(
            {
                "documents": "\n\n".join(state["documents"]),
                "generation": state["generation"],
            }
        ),
    )

    if hall_result.binary_score == "no":
        if retries < MAX_RETRIES:
            return "regenerate"
        return "web_search"  # fallback after max retries

    # --- Answer quality check ---
    ans_grader = ANSWER_GRADER_PROMPT | llm.with_structured_output(GradeAnswer)
    ans_result = cast(
        GradeAnswer,
        ans_grader.invoke(
            {
                "question": state["question"],
                "generation": state["generation"],
            }
        ),
    )

    if ans_result.binary_score == "yes":
        return "useful"

    if retries < MAX_RETRIES:
        return "web_search"

    return "useful"  # accept best-effort after max retries


# ---------------------------------------------------------------------------
# Node: increment_retries (helper called before regenerate)
# ---------------------------------------------------------------------------


def increment_retries(state: GraphState) -> dict[str, Any]:
    """Increments the retry counter and logs the step."""
    return {
        "retries": state.get("retries", 0) + 1,
        "steps": ["🔄 Retrying generation due to quality issues"],
    }
