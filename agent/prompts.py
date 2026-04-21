from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Router: decides whether to hit the vector store or do a live web search
# ---------------------------------------------------------------------------
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at routing questions about the EU AI Act and AI regulation.

Route the user question to the most appropriate data source:

- 'vectorstore' → questions about EU AI Act articles, recitals, risk categories,
  prohibited practices, compliance obligations, definitions, timelines in the act itself.

- 'web_search' → questions about recent news, implementation updates after 2024,
  national transposition laws, or enforcement decisions not covered in the document.

Return ONLY the single word: 'vectorstore' or 'web_search'. No other text.""",
        ),
        ("human", "Question: {question}"),
    ]
)

# ---------------------------------------------------------------------------
# Document relevance grader
# ---------------------------------------------------------------------------
DOC_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing the relevance of a retrieved document chunk
to a user question about the EU AI Act.

Score 'yes' if the document contains keywords, concepts, or semantic meaning
that is useful for answering the question.
Score 'no' if the document is clearly unrelated.

Return ONLY 'yes' or 'no'.""",
        ),
        ("human", "Document:\n\n{document}\n\nQuestion: {question}"),
    ]
)

# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert EU AI Act compliance advisor.

Use ONLY the context provided below to answer the question. 
Structure your answer clearly. If the context does not contain enough information,
say so explicitly — do not fabricate details.

When referencing specific requirements, mention the relevant Article or Recital
if identifiable from the context.

Context:
{context}""",
        ),
        ("human", "Question: {question}"),
    ]
)

# ---------------------------------------------------------------------------
# Hallucination grader: checks if generation is grounded in retrieved docs
# ---------------------------------------------------------------------------
HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing whether an LLM-generated answer is
fully grounded in the provided source documents.

Score 'yes' if every factual claim in the generation can be traced back
to the provided documents.
Score 'no' if the generation contains claims not supported by the documents.

Return ONLY 'yes' or 'no'.""",
        ),
        ("human", "Source documents:\n\n{documents}\n\nGeneration: {generation}"),
    ]
)

# ---------------------------------------------------------------------------
# Answer quality grader: checks if generation actually resolves the question
# ---------------------------------------------------------------------------
ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing whether an answer fully resolves a user's question.

Score 'yes' if the answer directly addresses the question and provides
useful information.
Score 'no' if the answer is evasive, off-topic, or incomplete.

Return ONLY 'yes' or 'no'.""",
        ),
        ("human", "Question: {question}\n\nAnswer: {generation}"),
    ]
)
