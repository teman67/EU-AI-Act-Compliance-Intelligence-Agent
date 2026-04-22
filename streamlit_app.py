"""
streamlit_app.py — Chat interface for the EU AI Act RAG agent.

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
    .step-badge {
        background: #eef2ff;
        border-left: 3px solid #4f46e5;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.85rem;
        color: #312e81;
    }
    .source-badge {
        background: #f0fdf4;
        border-left: 3px solid #16a34a;
        padding: 5px 12px;
        border-radius: 4px;
        margin: 3px 0;
        font-size: 0.82rem;
        color: #14532d;
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
        "- 🔄 Self-correcting retries\n"
        "- 📖 Source citation tracking"
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
            st.session_state["pending_question"] = q
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown(
        "Built with [LangGraph](https://langchain-ai.github.io/langgraph/) · "
        "[ChromaDB](https://www.trychroma.com/) · "
        "[OpenAI](https://openai.com/)"
    )

    st.markdown(
        "Source code: "
        "[GitHub](https://github.com/teman67/EU-AI-Act-Compliance-Intelligence-Agent)"
    )

    st.markdown("Developer: [Amirhossein Bayani](https://www.linkedin.com/in/amirhosseinbayani/)")

    st.markdown(
        "Disclaimer: This agent provides general information about the EU AI Act but does not constitute legal advice. "
        "For specific compliance questions, consult a qualified legal professional."
    )


# ---------------------------------------------------------------------------
# Prerequisites check
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


if not check_setup():
    st.stop()

# Import agent only after setup check passes
from agent.graph import rag_agent  # noqa: E402

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# ⚖️ EU AI Act Compliance Intelligence")
st.markdown(
    "Ask anything about the EU Artificial Intelligence Act. "
    "Powered by a self-correcting LangGraph RAG agent with adaptive retrieval and source citations."
)
st.divider()


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------
def run_agent(question: str) -> tuple[dict, float]:
    start = time.perf_counter()
    initial_state = {
        "question": question,
        "generation": "",
        "web_search": "No",
        "documents": [],
        "sources": [],
        "steps": [],
        "retries": 0,
    }
    result = rag_agent.invoke(initial_state)
    latency = round(time.perf_counter() - start, 2)
    return result, latency


# ---------------------------------------------------------------------------
# Message renderer
# ---------------------------------------------------------------------------
def render_assistant_message(msg: dict) -> None:
    """Render assistant bubble: answer, metrics, reasoning steps, and citations."""
    st.markdown(msg["content"])

    c1, c2, c3 = st.columns(3)
    c1.metric("⏱️ Latency", f"{msg.get('latency', 0)}s")
    c2.metric("📄 Docs Used", msg.get("docs_used", 0))
    c3.metric("🌐 Web Search", "Yes" if msg.get("web_search_used") else "No")

    steps = msg.get("steps", [])
    if steps:
        with st.expander("🔍 Reasoning Steps"):
            for step in steps:
                st.markdown(
                    f'<div class="step-badge">{step}</div>',
                    unsafe_allow_html=True,
                )

    sources = msg.get("sources", [])
    if sources:
        # Deduplicate sources by (source-type, url/page) key
        seen: set = set()
        unique_sources = []
        for src in sources:
            if src.get("source") == "web":
                key = ("web", src.get("url", ""))
            else:
                key = ("doc", src.get("page", ""))
            if key not in seen:
                seen.add(key)
                unique_sources.append(src)
        sources = unique_sources
    if sources:
        with st.expander(f"📖 Source Citations ({len(sources)})"):
            for i, src in enumerate(sources, 1):
                if src.get("source") == "web":
                    url = src.get("url", "")
                    title = src.get("title") or url or "Web result"
                    link = f'<a href="{url}" target="_blank">{title}</a>' if url else title
                    st.markdown(
                        f'<div class="source-badge">🌐 Chunk {i} — {link}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    page = src.get("page")
                    page_display = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
                    st.markdown(
                        f'<div class="source-badge">'
                        f"📄 Chunk {i} — {page_display}"
                        f" · EU AI Act (Official Journal L 2024/1689)"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Chat history init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render existing conversation
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚖️"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant_message(msg)

# ---------------------------------------------------------------------------
# Capture new question — sidebar button or typed input
# ---------------------------------------------------------------------------
question: str | None = None

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if typed := st.chat_input("Ask anything about the EU AI Act..."):
    question = typed

# ---------------------------------------------------------------------------
# Process question
# ---------------------------------------------------------------------------
if question:
    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    # Run agent inside a spinner; append result to session state and rerun so
    # the messages loop is the single render path (prevents double rendering).
    with st.chat_message("assistant", avatar="⚖️"), st.spinner("Agent is thinking..."):
        try:
            result, latency = run_agent(question)
            assistant_msg: dict = {
                "role": "assistant",
                "content": result.get("generation", "No answer generated."),
                "steps": result.get("steps", []),
                "sources": result.get("sources", []),
                "web_search_used": result.get("web_search") == "Yes",
                "docs_used": len(result.get("documents", [])),
                "latency": latency,
            }
        except Exception as e:
            assistant_msg = {
                "role": "assistant",
                "content": f"❌ Agent error: {e}",
                "steps": [],
                "sources": [],
                "web_search_used": False,
                "docs_used": 0,
                "latency": 0.0,
            }

    st.session_state["messages"].append(assistant_msg)
    st.rerun()
