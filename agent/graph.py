"""
graph.py — Builds and compiles the EU AI Act RAG agent as a LangGraph StateGraph.

Graph flow (Adaptive + Corrective RAG pattern):

                        ┌─────────────┐
                        │   START     │
                        └──────┬──────┘
                               │
                    ┌──────────▼──────────┐
                    │   route_question    │  (conditional: vectorstore / web_search)
                    └──────────┬──────────┘
                   ┌───────────┴────────────┐
                   ▼                        ▼
            ┌─────────┐             ┌────────────┐
            │retrieve │             │ web_search │
            └────┬────┘             └─────┬──────┘
                 │                        │
         ┌───────▼────────┐               │
         │grade_documents │               │
         └───────┬────────┘               │
        ┌────────┴─────────┐              │
   relevant?              few docs?       │
        │                  └──► web_search┘
        ▼                        │
   ┌──────────┐◄─────────────────┘
   │ generate │◄──────────────────────┐
   └────┬─────┘                       │  (regenerate)
        │                             │
  ┌─────▼──────────────┐              │
  │  grade_generation  │──► increment_retries ──┘
  └─────┬──────────────┘
        │ useful
        ▼
      END
"""

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    generate,
    grade_documents,
    grade_generation,
    handle_off_topic,
    increment_retries,
    retrieve,
    route_question,
    web_search,
)
from agent.state import GraphState


def build_graph():
    """Construct and compile the RAG agent graph."""

    graph = StateGraph(GraphState)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------
    graph.add_node("retrieve", retrieve)
    graph.add_node("web_search_node", web_search)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("increment_retries", increment_retries)
    graph.add_node("handle_off_topic", handle_off_topic)

    # ------------------------------------------------------------------
    # Entry point: conditional routing from START
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        START,
        route_question,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search_node",
            "off_topic": "handle_off_topic",
        },
    )

    # ------------------------------------------------------------------
    # Off-topic questions go straight to END
    # ------------------------------------------------------------------
    graph.add_edge("handle_off_topic", END)

    # ------------------------------------------------------------------
    # After retrieval → grade documents
    # ------------------------------------------------------------------
    graph.add_edge("retrieve", "grade_documents")

    # ------------------------------------------------------------------
    # After grading → generate (enough docs) or web_search (too few)
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: "web_search" if state.get("web_search") == "Yes" else "generate",
        {
            "web_search": "web_search_node",
            "generate": "generate",
        },
    )

    # ------------------------------------------------------------------
    # Web search always leads to generation
    # ------------------------------------------------------------------
    graph.add_edge("web_search_node", "generate")

    # ------------------------------------------------------------------
    # After generation → quality check (conditional)
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "useful": END,
            "regenerate": "increment_retries",
            "web_search": "web_search_node",
        },
    )

    # ------------------------------------------------------------------
    # After incrementing retries → retry generation
    # ------------------------------------------------------------------
    graph.add_edge("increment_retries", "generate")

    return graph.compile()


# Compile once at module level for import
rag_agent = build_graph()
