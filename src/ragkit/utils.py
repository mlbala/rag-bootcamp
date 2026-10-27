"""Helpers for looking at Documents, chunks and retriever results while building a RAG pipeline."""

from collections import Counter
from pathlib import Path
from statistics import mean

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .config import PROJECT_ROOT


def short_source(doc: Document) -> str:
    """The document's "source", relative to the project root when it is a file inside the project.

    Loaders given ragkit.config paths store absolute paths; this turns
    "/Users/.../RAG-Bootcamp/data/raw/text/ai_agents.txt" back into "data/raw/text/ai_agents.txt".
    Anything else (a URL, a table name, "notes") is returned unchanged.
    """
    source = str(doc.metadata.get("source", "unknown"))
    path = Path(source)
    if path.is_absolute() and path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return source


def source_label(doc: Document) -> str:
    """Short source plus page, sheet, table or row when there is one, e.g. "data/raw/pdf/1706.03762v7.pdf p.3"."""
    meta = doc.metadata
    label = short_source(doc)
    if "page" in meta:
        label += f" p.{meta['page'] + 1}"   # PyPDFLoader counts pages from 0
    if "sheet" in meta:
        label += f" sheet {meta['sheet']}"
    if "table" in meta:
        label += f" table {meta['table']}"
    if "row" in meta:
        label += f" row {meta['row']}"
    return label


def format_docs(docs: list[Document]) -> str:
    """Join retrieved chunks into one context string for a prompt, each labelled with its source."""
    return "\n\n".join(f"[Source: {source_label(doc)}]\n{doc.page_content}" for doc in docs)


def preview(text: str, max_chars: int = 200) -> str:
    """The text on one line (newlines and repeated spaces collapsed), cut to max_chars."""
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def print_docs(
    docs: list[Document] | list[tuple[Document, float]],
    preview_chars: int = 200,
    show_metadata: bool = True,
) -> None:
    """Print each Document's source, length, metadata and a preview of its text.

    Also accepts the (Document, score) pairs returned by similarity_search_with_score().
    """
    for i, item in enumerate(docs, start=1):
        doc, score = item if isinstance(item, tuple) else (item, None)
        header = f"--- {i} | {source_label(doc)} | {len(doc.page_content)} chars"
        if score is not None:
            header += f" | score: {score:.4f}"
        print(header + " ---")
        if show_metadata:
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
        print(f"  {preview(doc.page_content, preview_chars)}\n")


def inspect_chunks(chunks: list[Document], show: int = 3, preview_chars: int = 150) -> dict:
    """Summarise a split: chunk count, size range, chunks per source, then the first `show` chunks.

    Returns the numbers as a dict, so they can be compared across chunk_size / chunk_overlap settings.
    """
    if not chunks:
        print("No chunks")
        return {"count": 0}

    lengths = [len(chunk.page_content) for chunk in chunks]
    per_source = Counter(short_source(chunk) for chunk in chunks)
    stats = {
        "count": len(chunks),
        "min_chars": min(lengths),
        "avg_chars": round(mean(lengths)),
        "max_chars": max(lengths),
        "per_source": dict(per_source),
    }

    print(f"{stats['count']} chunks | characters: min {stats['min_chars']}, "
          f"avg {stats['avg_chars']}, max {stats['max_chars']}")
    print("Chunks per source:")
    for source, count in per_source.most_common():
        print(f"  {count:4}  {source}")
    if show:
        print(f"\nFirst {min(show, len(chunks))} chunks:")
        print_docs(chunks[:show], preview_chars=preview_chars)
    return stats


def compare_retrievers(
    query: str,
    retrievers: dict[str, BaseRetriever],
    preview_chars: int = 100,
) -> dict[str, list[Document]]:
    """Run one query through several retrievers and print what each one returned.

    Chunks marked * were returned by that retriever only, which makes it easy to see what
    MMR vs similarity, dense vs BM25, or a different k / filter actually changes.
    Returns {retriever name: [Document, ...]}.
    """
    results = {name: retriever.invoke(query) for name, retriever in retrievers.items()}

    # Match chunks by content, not id: BM25 and other non-vector retrievers return Documents without one
    def key(doc):
        return short_source(doc), doc.page_content

    found_by = Counter(k for docs in results.values() for k in {key(doc) for doc in docs})

    print(f"Query: {query}\n")
    for name, docs in results.items():
        print(f"===== {name}: {len(docs)} results =====")
        for i, doc in enumerate(docs, start=1):
            marker = "*" if found_by[key(doc)] == 1 and len(results) > 1 else " "
            print(f" {marker}{i}. {source_label(doc)} | {preview(doc.page_content, preview_chars)}")
        print()

    shared = sum(count == len(results) for count in found_by.values())
    print(f"{shared} chunk(s) returned by all {len(results)} retrievers, {len(found_by)} distinct chunks in total")
    return results
