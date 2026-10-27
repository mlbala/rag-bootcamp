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

```text
RAG-Bootcamp/
├── .env.example                      Every API key the notebooks read; copy it to .env (gitignored)
├── pyproject.toml, uv.lock           Dependencies; `uv sync` also installs src/ragkit/ in editable mode
│
├── data/
│   ├── data_parsing/                 Parsed documents
│   ├── processed/                    Generated files (gitignored, the folder is kept)
│   └── raw/                          Source documents used by the notebooks
│       ├── databases/                  SQLite (company.db)
│       ├── json_files/                 JSON and JSON Lines
│       ├── pdf/                        Research papers
│       ├── structured_files/           CSV and Excel
│       ├── text/                       Plain-text articles
│       └── word_files/                 .docx
│
├── vectorstore/                      Local vector stores built by the notebooks (gitignored)
│   ├── chroma_db/
│   └── faiss_index/
│
├── src/
│   └── ragkit/                       Shared code for notebooks and apps, imported as `ragkit`
│       ├── __init__.py
│       ├── config.py                   Project paths, and loads .env
│       ├── llm_factory.py              get_llm() and get_embeddings()
│       └── utils.py                    Document, chunk and retriever inspection helpers
│
├── notebooks/
│   ├── 01_ingestion_and_chunking/
│   │   ├── 01_data_ingestion_parsing/    Text, PDF, Word, CSV/Excel, JSON and database loaders
│   │   ├── 02_vector_embeddings/         HuggingFace and OpenAI embeddings, semantic similarity
│   │   ├── 03_vector_stores_databases/   Chroma (local and Cloud), FAISS, in-memory, Astra DB, Pinecone, Milvus
│   │   └── 04_advanced_chunking/
│   ├── 02_advanced_retrieval/
│   │   └── 01_hybrid_search/
│   ├── 03_multimodal_and_graph_rag/
│   ├── 04_agentic_rag_langgraph/
│   └── 05_eval_and_observability/
│
└── apps/
    ├── domain_knowledge_chatbot/
    ├── multimodal_assistant/
    └── multi_agent_report_generator/
```

`data/` and `vectorstore/` live at the project root, so notebooks, `ragkit` and apps all read and write the same files.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # installs the dependencies, and src/ragkit/ as the editable package `ragkit`
cp .env.example .env     # then fill in the keys for the notebooks you run
```

Then open a notebook in VS Code or Jupyter using the `.venv` kernel.

## Running the notebooks

- Notebooks take their paths from `ragkit.config` (`RAW_TEXT_DIR`, `CHROMA_PATH`, ...), so each one runs from its own folder: there is no `os.chdir` and no relative `data/...` path.
- API keys are read from `.env` in the project root. `.env.example` lists the keys each notebook needs.
- Local vector stores are written to `vectorstore/`. `01_chromadb` rebuilds `chroma_db/` on every run and `02_conversational_memory` reuses it, so run `01_chromadb` first. `03_faiss` builds `faiss_index/`.
- The cloud notebooks (Chroma Cloud, Astra DB, Pinecone, Milvus) upsert chunks with stable ids, so re-running them overwrites the same chunks instead of duplicating them.

**`ModuleNotFoundError: No module named 'ragkit'`?** The kernel was started before `uv sync`, or it isn't the `.venv` interpreter: select the `.venv` kernel and restart it. If a notebook was open while files were moved or edited on disk (it still imports from the old `src` package), close its tab without saving and reopen it.

## Shared code: the `ragkit` package (`src/ragkit/`)

The repo uses the standard src layout: `uv sync` installs `src/ragkit/` as the editable package `ragkit`, so `from ragkit... import ...` works from any folder (notebooks, apps, scripts) without `sys.path` changes or `os.chdir`, and edits to `src/ragkit/` are picked up on the next kernel restart. Only `src/` is put on `sys.path`, not the project root, so folders like `data/` or `notebooks/` can't be imported by accident.

### Paths: `ragkit.config`

Importing `ragkit.config` also loads `.env` from the project root. Every constant is an absolute `pathlib.Path`:

| Constant | Folder |
| --- | --- |
| `PROJECT_ROOT` | the repo root |
| `DATA_DIR` | `data/` |
| `DATA_PARSING_DIR` | `data/data_parsing/` |
| `PROCESSED_DATA_DIR` | `data/processed/` |
| `RAW_DATA_DIR` | `data/raw/` |
| `RAW_DB_DIR` | `data/raw/databases/` |
| `RAW_JSON_DIR` | `data/raw/json_files/` |
| `RAW_PDF_DIR` | `data/raw/pdf/` |
| `RAW_STRUCTURED_DIR` | `data/raw/structured_files/` |
| `RAW_TEXT_DIR` | `data/raw/text/` |
| `RAW_WORD_DIR` | `data/raw/word_files/` |
| `VECTOR_STORE_DIR` | `vectorstore/` |
| `CHROMA_PATH` | `vectorstore/chroma_db/` |
| `FAISS_PATH` | `vectorstore/faiss_index/` |

```python
from ragkit.config import RAW_PDF_DIR, CHROMA_PATH, FAISS_PATH

docs = PyPDFLoader(RAW_PDF_DIR / "1706.03762v7.pdf").load()
faiss_store.save_local(FAISS_PATH)
chroma_store = Chroma(collection_name="text_docs", embedding_function=embeddings,
                      persist_directory=str(CHROMA_PATH))   # Chroma needs a str, not a Path
```

### Models: `ragkit.llm_factory`

```python
from ragkit.llm_factory import get_llm, get_embeddings

llm = get_llm()                                   # OpenAI gpt-5, temperature 0
llm = get_llm("groq")                             # openai/gpt-oss-120b on Groq (GROQ_API_KEY)
llm = get_llm("ollama", model="llama3.2")         # a local Ollama server
embeddings = get_embeddings()                     # OpenAI text-embedding-3-small
embeddings = get_embeddings("huggingface")        # sentence-transformers/all-MiniLM-L6-v2, runs locally
```

Each provider's package is imported only when it is used, and extra keyword arguments go to the model class (`get_embeddings("openai", dimensions=1024)`).

### Inspection: `ragkit.utils`

| Helper | What it does |
| --- | --- |
| `inspect_chunks(chunks)` | Chunk count, size range, chunks per source, and the first few chunks |
| `print_docs(docs)` | Source, length, metadata and a preview of each Document; also takes `(doc, score)` pairs |
| `compare_retrievers(query, {"name": retriever, ...})` | Runs one query through several retrievers and marks the chunks only one of them found |
| `format_docs(docs)` | Joins chunks into a prompt context, each labelled `[Source: ...]` |
| `source_label(doc)` | `data/raw/pdf/1706.03762v7.pdf p.3`: the project-relative source plus page, sheet, table or row |

Loaders given `ragkit.config` paths store absolute paths in `metadata["source"]`. `source_label()` and `format_docs()` shorten them to project-relative paths.
