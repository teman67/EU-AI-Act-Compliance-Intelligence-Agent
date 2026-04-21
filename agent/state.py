import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    """
    Represents the state flowing through the RAG agent graph.

    Attributes:
        question:   The user's input question.
        generation: The LLM's generated answer.
        web_search: Whether a web search was triggered ("Yes" / "No").
        documents:  List of retrieved document chunks (as strings).
        steps:      Accumulated list of agent steps for tracing/UI display.
        retries:    Number of generation retry attempts made.
    """

    question: str
    generation: str
    web_search: str
    documents: list[str]
    steps: Annotated[list[str], operator.add]
    retries: int
