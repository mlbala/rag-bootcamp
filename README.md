# RAG Bootcamp

**The Ultimate RAG Bootcamp: From Traditional to Advanced Agentic AI Systems.** A hands-on tutorial on Retrieval-Augmented Generation (RAG), from the basics through to production-ready agentic pipelines, using LangChain, LangGraph, and LangSmith.

## What this tutorial covers

1. **RAG foundations**: ingestion, parsing, embeddings, vector databases, and retrieval + generation with LangChain.
2. **Advanced RAG**: advanced chunking, hybrid search, multimodal RAG, persistent memory, Self-RAG, and Adaptive & Corrective RAG.
3. **Agentic RAG**: multi-agent pipelines with LangGraph for research, summarization, and decision-making.
4. **Evaluation & optimization**: tracking, debugging, and evaluating RAG systems with LangSmith.
5. **Real-world projects**: a domain-specific chatbot, a multi-agent research assistant, a multimodal assistant, and cloud deployment.

**Tools:** LangChain, LangGraph, LangSmith, FAISS, ChromaDB, Pinecone, Weaviate.

## Repo layout

```
notebooks/   Step-by-step course notebooks
src/ragkit/  Reusable helper code
data/        Raw documents, parsed output, and local vector stores
```

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file in the project root with the API keys you need (for example `OPENAI_API_KEY`, `GROQ_API_KEY`, `LANGSMITH_API_KEY`). It is gitignored.

Then open a notebook in VS Code or Jupyter using the `.venv` kernel.
