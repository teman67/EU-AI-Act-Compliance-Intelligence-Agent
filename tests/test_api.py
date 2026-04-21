from types import SimpleNamespace

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == "1.0.0"
    assert response.headers.get("X-Request-ID")


def test_query_rejects_blank_question() -> None:
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422
    assert response.json()["detail"] == "Question cannot be empty."


def test_query_success(monkeypatch) -> None:
    def fake_invoke(state):
        assert state["question"] == "What is a high-risk AI system?"
        return {
            "generation": "High-risk AI systems are subject to strict obligations.",
            "steps": ["retrieved", "generated"],
            "web_search": "No",
        }

    monkeypatch.setattr(api, "rag_agent", SimpleNamespace(invoke=fake_invoke))

    response = client.post(
        "/query",
        json={"question": "What is a high-risk AI system?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("High-risk AI systems")
    assert payload["web_search_used"] is False
    assert payload["steps"] == ["retrieved", "generated"]
    assert response.headers.get("X-Request-ID")


def test_query_surfaces_agent_error(monkeypatch) -> None:
    def fake_invoke(_state):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(api, "rag_agent", SimpleNamespace(invoke=fake_invoke))

    response = client.post(
        "/query",
        json={"question": "What is prohibited AI?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error. Please try again later."
    assert response.headers.get("X-Request-ID")
