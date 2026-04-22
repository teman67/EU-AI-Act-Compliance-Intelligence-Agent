from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from agent import nodes
from agent.state import GraphState


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class PassthroughPrompt:
    def __or__(self, rhs: Any) -> Any:
        return rhs


class FakeStructuredInvoker:
    def __init__(self, response: Any):
        self.response = response

    def invoke(self, _payload: dict[str, Any]) -> Any:
        return self.response


class FakeLLM:
    def __init__(self, responses: dict[str, Any]):
        self.responses = responses

    def with_structured_output(self, schema: type[Any]) -> FakeStructuredInvoker:
        return FakeStructuredInvoker(self.responses[schema.__name__])


def make_state(**overrides: Any) -> GraphState:
    state: dict[str, Any] = {
        "question": "What are provider obligations?",
        "generation": "",
        "web_search": "No",
        "documents": [],
        "steps": [],
        "retries": 0,
    }
    state.update(overrides)
    return cast(GraphState, state)


def test_retrieve_maps_docs_to_text(monkeypatch) -> None:
    class FakeRetriever:
        def invoke(self, _question: str) -> list[FakeDoc]:
            return [FakeDoc("doc-a"), FakeDoc("doc-b")]

    monkeypatch.setattr(nodes, "_get_retriever", lambda: FakeRetriever())

    result = nodes.retrieve(make_state())

    assert result["documents"] == ["doc-a", "doc-b"]
    assert result["sources"] == [{}, {}]
    assert result["steps"] == ["🔍 Retrieved documents from vector store"]


def test_web_search_appends_results(monkeypatch) -> None:
    class FakeSearch:
        def invoke(self, _payload: dict[str, Any]) -> list[dict[str, str]]:
            return [{"content": "web-1"}, {"content": "web-2"}]

    monkeypatch.setattr(nodes, "_get_web_search_tool", lambda: FakeSearch())

    result = nodes.web_search(make_state(documents=["existing-doc"]))

    assert result["documents"] == ["existing-doc", "web-1", "web-2"]
    assert result["web_search"] == "Yes"


def test_grade_documents_filters_and_triggers_web_search(monkeypatch) -> None:
    class ContentBasedDocGrader:
        def invoke(self, payload: dict[str, str]) -> nodes.GradeDocuments:
            score = "yes" if "relevant" in payload["document"] else "no"
            return nodes.GradeDocuments(binary_score=score)

    class FakeDocLLM:
        def with_structured_output(self, _schema: type[Any]) -> ContentBasedDocGrader:
            return ContentBasedDocGrader()

    monkeypatch.setattr(nodes, "DOC_GRADER_PROMPT", PassthroughPrompt())
    monkeypatch.setattr(nodes, "_get_llm", lambda: FakeDocLLM())

    result = nodes.grade_documents(make_state(documents=["relevant article", "off-topic text"]))

    assert result["documents"] == ["relevant article"]
    assert result["web_search"] == "Yes"


def test_grade_generation_requests_regeneration_when_hallucinated(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "HALLUCINATION_GRADER_PROMPT", PassthroughPrompt())
    monkeypatch.setattr(nodes, "ANSWER_GRADER_PROMPT", PassthroughPrompt())
    monkeypatch.setattr(
        nodes,
        "_get_llm",
        lambda: FakeLLM(
            {
                "GradeHallucinations": nodes.GradeHallucinations(binary_score="no"),
                "GradeAnswer": nodes.GradeAnswer(binary_score="yes"),
            }
        ),
    )

    decision = nodes.grade_generation(make_state(documents=["doc"], generation="answer", retries=0))

    assert decision == "regenerate"


def test_grade_generation_uses_web_search_after_max_hallucination_retries(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "HALLUCINATION_GRADER_PROMPT", PassthroughPrompt())
    monkeypatch.setattr(nodes, "ANSWER_GRADER_PROMPT", PassthroughPrompt())
    monkeypatch.setattr(
        nodes,
        "_get_llm",
        lambda: FakeLLM(
            {
                "GradeHallucinations": nodes.GradeHallucinations(binary_score="no"),
                "GradeAnswer": nodes.GradeAnswer(binary_score="yes"),
            }
        ),
    )

    decision = nodes.grade_generation(make_state(documents=["doc"], generation="answer", retries=2))

    assert decision == "web_search"


def test_grade_generation_marks_useful_when_grounded_and_answered(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "HALLUCINATION_GRADER_PROMPT", PassthroughPrompt())
    monkeypatch.setattr(nodes, "ANSWER_GRADER_PROMPT", PassthroughPrompt())
    monkeypatch.setattr(
        nodes,
        "_get_llm",
        lambda: FakeLLM(
            {
                "GradeHallucinations": nodes.GradeHallucinations(binary_score="yes"),
                "GradeAnswer": nodes.GradeAnswer(binary_score="yes"),
            }
        ),
    )

    decision = nodes.grade_generation(make_state(documents=["doc"], generation="answer", retries=0))

    assert decision == "useful"


def test_increment_retries_increases_counter() -> None:
    result = nodes.increment_retries(make_state(retries=1))

    assert result["retries"] == 2
    assert result["steps"] == ["🔄 Retrying generation due to quality issues"]
