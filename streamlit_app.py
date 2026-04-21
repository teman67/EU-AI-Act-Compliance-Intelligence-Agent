"""
streamlit_app.py — Interactive demo UI for the EU AI Act RAG agent.

Run with:
    streamlit run streamlit_app.py
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EU AI Act Compliance Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3c6e;
    }
    .sub-header {
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .step-badge {
        background: #eef2ff;
        border-left: 3px solid #4f46e5;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.85rem;
        color: #312e81;
    }
    .answer-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 16px;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg", width=60)
    st.markdown("## ⚖️ EU AI Act Agent")
    st.markdown(
        "An **Adaptive RAG** agent built with LangGraph that answers "
        "questions about the EU Artificial Intelligence Act.\n\n"
        "**Architecture:**\n"
        "- 🔀 Adaptive query routing\n"
        "- 📋 Document relevance grading\n"
        "- 🌐 Web search fallback\n"
        "- 🛡️ Hallucination detection\n"
        "- 🔄 Self-correcting retries"
    )

    st.divider()
    st.markdown("**Example questions:**")

    example_questions = [
        "What are the four risk categories in the EU AI Act?",
        "Which AI applications are prohibited under the Act?",
        "What obligations apply to providers of high-risk AI systems?",
        "What is the role of the AI Office?",
        "How does the Act define a general-purpose AI model?",
        "What are the transparency requirements for AI systems interacting with humans?",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill"] = q

    st.divider()
    st.markdown(
        "Built with [LangGraph](https://langchain-ai.github.io/langgraph/) · "
        "[ChromaDB](https://www.trychroma.com/) · "
        "[OpenAI](https://openai.com/)"
    )


# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
def check_setup() -> bool:
    issues = []
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("❌ `OPENAI_API_KEY` not set in `.env`")
    if not Path("./chroma_db").exists():
        issues.append("❌ Vector store not built — run `python ingest.py` first")
    if issues:
        for issue in issues:
            st.error(issue)
        return False
    return True


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">⚖️ EU AI Act Compliance Intelligence</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Ask anything about the EU Artificial Intelligence Act. '
    "Powered by a self-correcting LangGraph RAG agent with adaptive retrieval.</div>",
    unsafe_allow_html=True,
)

if not check_setup():
    st.stop()

# Import agent only after setup check passes
from agent.graph import rag_agent  # noqa: E402

# Pre-fill from sidebar button click
default_q = st.session_state.pop("prefill", "")

question = st.text_area(
    "Your question:",
    value=default_q,
    placeholder="e.g. What obligations do providers of high-risk AI systems have?",
    height=80,
)

col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Ask Agent ▶", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------
if run and question.strip():
    with st.spinner("Agent is thinking..."):
        start = time.perf_counter()

        initial_state = {
            "question": question,
            "generation": "",
            "web_search": "No",
            "documents": [],
            "steps": [],
            "retries": 0,
        }

        try:
            result = rag_agent.invoke(initial_state)
            latency = round(time.perf_counter() - start, 2)

            # --- Metrics row ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("⏱️ Latency", f"{latency}s")
            with m2:
                st.metric("📄 Documents Used", len(result.get("documents", [])))
            with m3:
                web_used = result.get("web_search") == "Yes"
                st.metric("🌐 Web Search", "Yes" if web_used else "No")

            st.divider()

            # --- Agent trace ---
            with st.expander("🔍 Agent Reasoning Steps", expanded=True):
                for step in result.get("steps", []):
                    st.markdown(
                        f'<div class="step-badge">{step}</div>',
                        unsafe_allow_html=True,
                    )

            # --- Answer ---
            st.markdown("### Answer")
            st.markdown(
                f'<div class="answer-box">{result.get("generation", "No answer generated.")}</div>',
                unsafe_allow_html=True,
            )

            # --- Source chunks ---
            docs = result.get("documents", [])
            if docs:
                with st.expander(f"📚 Source Chunks Used ({len(docs)})"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc[:500] + ("..." if len(doc) > 500 else ""))
                        st.divider()

        except Exception as e:
            st.error(f"Agent error: {e}")

elif run and not question.strip():
    st.warning("Please enter a question.")
