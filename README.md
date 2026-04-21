# ⚖️ EU AI Act Compliance Intelligence Agent

![CI](https://github.com/teman67/EU-AI-Act-Compliance-Intelligence-Agent/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)
![License](https://img.shields.io/badge/license-MIT-green)

A portfolio-ready **Adaptive RAG agent** built with **LangGraph** and **ChromaDB** that answers questions about the EU Artificial Intelligence Act using self-correcting retrieval, document grading, hallucination detection, and **source citation tracking**.

> **Live Demo** · Not deployed yet (run locally) &nbsp;|&nbsp; **API Docs** · `/docs` when running FastAPI

---

## Architecture

The agent implements the **Adaptive + Corrective RAG** pattern — one of the most robust LangGraph architectures for production RAG systems.

```
                     ┌─────────────────┐
                     │   User Query    │
                     └────────┬────────┘
                              │
               ┌──────────────▼──────────────┐
               │      route_question          │
               │  (LLM-based query router)    │
               └──────┬───────────────┬───────┘
                      │               │
               vectorstore        web_search
                      │               │
              ┌───────▼──────┐        │
              │   retrieve   │        │
              │  (ChromaDB)  │        │
              └───────┬──────┘        │
                      │               │
          ┌───────────▼────────────┐  │
          │    grade_documents     │  │
          │  (relevance filtering) │  │
          └───────┬────────────────┘  │
          relevant│  too few↓         │
                  │    web_search ◄───┘
                  │         │
              ┌───▼─────────▼──┐
              │    generate    │ ◄─────────┐
              │  (RAG prompt)  │           │
              └───────┬────────┘           │
                      │                    │ regenerate
          ┌───────────▼──────────────┐     │
          │     grade_generation      │     │
          │  hallucination + quality  │─────┘
          └───────────┬──────────────┘
                      │ useful
                   ┌──▼──┐
                   │ END │
                   └─────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over plain LangChain | Explicit state machine enables retry loops and conditional routing impossible with chains |
| Document grading before generation | Filters noise early, improves answer quality significantly |
| Hallucination detection post-generation | Catches unsupported claims before they reach the user |
| Tavily web search fallback | Handles questions about recent enforcement decisions not in the static document |
| ChromaDB with persistence | Local vector store with no external dependencies for development |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | **LangGraph 0.2** |
| LLM | **OpenAI GPT-5.4-mini** |
| Embeddings | **OpenAI text-embedding-3-small** |
| Vector Store | **ChromaDB** |
| Web Search | **Tavily** |
| Backend API | **FastAPI** |
| Frontend Demo | **Streamlit** (chat interface) |
| Containerisation | **Docker / docker-compose** |

---

## Quickstart

### Docker (recommended)

```bash
# Build and run the Streamlit app
docker compose up --build

# App available at http://localhost:8501
```

> Before building, create a `.env` file with your API keys (see step 2).  
> Run `python ingest.py` once to populate `./chroma_db`, then mount it into the container via `docker compose up`.

### Manual setup

```bash
git clone https://github.com/teman67/EU-AI-Act-Compliance-Intelligence-Agent.git
cd EU-AI-Act-Compliance-Intelligence-Agent
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and TAVILY_API_KEY
```

Get a free Tavily key at [app.tavily.com](https://app.tavily.com).

Optional API hardening settings:

```bash
CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:8501
LOG_LEVEL=INFO
```

### 3. Build the vector store (run once)

```bash
python ingest.py
```

This downloads the official EU AI Act PDF from EUR-Lex, splits it into ~1,000-character chunks, embeds them with OpenAI, and persists the ChromaDB collection to `./chroma_db`.

### 4. Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

### 5. Or run the FastAPI backend

```bash
uvicorn api:app --reload
# API docs at http://localhost:8000/docs
```

### 6. Developer quality checks

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
mypy .
pytest -q
```

Optional local hooks:

```bash
pre-commit install
pre-commit run --all-files
```

---

## Example Queries

```
"What are the four risk categories defined in the EU AI Act?"
"Which AI applications are explicitly prohibited?"
"What technical documentation must providers of high-risk AI systems maintain?"
"How does the Act define a general-purpose AI model?"
"What are the transparency requirements for AI chatbots?"
"What fines can be imposed for violations?"
```

---

## API Reference

### `POST /query`

```json
{
  "question": "What obligations apply to deployers of high-risk AI systems?"
}
```

**Response:**

```json
{
  "question": "What obligations apply to deployers of high-risk AI systems?",
  "answer": "According to Article 26 of the EU AI Act, deployers of high-risk AI systems must...",
  "steps": [
    "🔍 Retrieved documents from vector store",
    "📋 Graded 4 documents → 3 relevant",
    "✍️  Generated answer (attempt 1)"
  ],
  "web_search_used": false,
  "latency_seconds": 3.21
}
```

Notes:
- API responses include an `X-Request-ID` header for request tracing.
- Internal server errors return a generic message (no raw exception leakage).

---

## Project Structure

```
eu-ai-act-agent/
├── agent/
│   ├── state.py        # GraphState TypedDict
│   ├── prompts.py      # All LangChain prompt templates
│   ├── nodes.py        # Node functions (retrieve, grade, generate...)
│   └── graph.py        # LangGraph StateGraph definition
├── ingest.py           # One-time vector store builder
├── api.py              # FastAPI REST API
├── streamlit_app.py    # Streamlit demo UI
├── tests/              # Unit and API tests
├── .github/workflows/  # CI pipeline
├── pyproject.toml      # Ruff, Mypy, Pytest config
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
 
```

---

## Deployment

### Docker (Streamlit Cloud / free hosting)
1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Add `OPENAI_API_KEY` and `TAVILY_API_KEY` in Streamlit secrets
4. Run `ingest.py` during your deploy flow and store vectors in a managed vector database for production use

---

## Extending This Project

- **Swap ChromaDB for Pinecone or Weaviate** for a hosted vector store
- **Add multi-document support** (upload national AI laws, GDPR, etc.)
- ~~**Add citation tracking**~~ ✅ Done — source page numbers returned with every answer
- **Add conversation memory** using LangGraph's `MemorySaver` checkpointer
- **Evaluate with RAGAS** — automated RAG evaluation pipeline

---

## License

MIT — see [LICENSE](LICENSE).
