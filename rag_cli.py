"""
CLI interface for building and querying the RGD RAG system.

Usage examples:
  python rag_cli.py --build
  python rag_cli.py --question "How do I search for a gene in RGD?"
  python rag_cli.py --interactive
  python rag_cli.py --inspect-pdfs
  python rag_cli.py --add papers/new_paper.pdf
  python rag_cli.py --remove RGDHelpMarkdown/glossary.md
  python rag_cli.py --list-docs
"""

import argparse
import os
from pathlib import Path

from RAG import RAGSystem

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]


def get_documents_from_folder(folder_path: str):
    """Return all supported documents in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(list(folder.glob(f"*{ext}")))
    return sorted([str(f) for f in files])


def build_index(rag: RAGSystem, folders):
    """Incrementally (re)build the index from one or more folders."""
    all_docs = []
    for folder in folders:
        all_docs.extend(get_documents_from_folder(folder))
    all_docs = sorted(set(all_docs))
    if not all_docs:
        print(f"No documents found in: {', '.join(folders)}")
        return False

    print(f"Found {len(all_docs)} document(s) across {len(folders)} folder(s).")
    counts = {"added": 0, "updated": 0, "skipped": 0, "missing": 0}
    for doc in all_docs:
        status = rag.add_or_update_document(doc)
        counts[status] = counts.get(status, 0) + 1
    print(
        f"Index built. Added: {counts['added']}, Updated: {counts['updated']}, "
        f"Skipped (unchanged): {counts['skipped']}. Total chunks: {rag.store.count()}"
    )
    return True


def show_context(rag: RAGSystem, question: str, n_results: int, min_similarity: float):
    retrieved = rag.retrieve(question, n_results=n_results, min_similarity=min_similarity)
    print("Retrieved context:")
    for i, doc in enumerate(retrieved, 1):
        meta = doc.get("metadata", {})
        filename = meta.get("filename", "Unknown")
        section = meta.get("section", "")
        sim = doc.get("similarity", 0)
        preview = doc.get("document", "")[:200].replace("\n", " ")
        label = f"{filename} — {section}" if section else filename
        print(f"[{i}] {label} (sim: {sim:.3f}) {preview}...")
    print()


def answer(rag: RAGSystem, question: str, args):
    if args.show_context:
        show_context(rag, question, args.n_results, args.min_similarity)
    result = rag.generate(
        question,
        n_results=args.n_results,
        min_similarity=args.min_similarity,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(f"Q: {question}")
    print(f"A: {result}")


def interactive_shell(rag: RAGSystem, args):
    print("Interactive RGD RAG shell. Type 'exit' to quit.")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        answer(rag, question, args)
        print()


def main():
    parser = argparse.ArgumentParser(description="RGD RAG CLI")
    parser.add_argument("--build", action="store_true", help="Incrementally (re)build the index")
    parser.add_argument("--papers-dir", default="RGDHelpMarkdown", help="Path to documents folder")
    parser.add_argument("--papers-dirs", nargs="+", help="List of document folders to index")
    parser.add_argument("--collection", default="rgd_chunks", help="Collection/table name")
    parser.add_argument("--vector-backend", choices=["pgvector", "chroma"], default="pgvector", help="Vector store backend")
    parser.add_argument("--pg-dsn", default=os.environ.get("PG_DSN"), help="PostgreSQL DSN (or set PG_DSN env)")
    parser.add_argument("--persist-dir", default="./chroma_db", help="ChromaDB / links dir")
    parser.add_argument("--embedding-host", default="http://grudge.rgd.mcw.edu:11434", help="Ollama embeddings URL")
    parser.add_argument("--llm-host", default="http://grudge.rgd.mcw.edu:11434", help="Ollama chat URL")
    parser.add_argument("--embedding-model", default="mxbai-embed-large:latest", help="Ollama embedding model")
    parser.add_argument("--llm-model", default="llama3.2", help="LLM model name")
    parser.add_argument("--question", help="Ask a single question")
    parser.add_argument("--interactive", action="store_true", help="Run interactive shell")
    parser.add_argument("--n-results", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Minimum similarity threshold")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--seed", type=int, default=None, help="LLM seed for deterministic output")
    parser.add_argument("--show-context", action="store_true", help="Show retrieved context chunks")
    # Document management
    parser.add_argument("--add", metavar="PATH", help="Add/update a single document")
    parser.add_argument("--remove", metavar="SOURCE", help="Remove a document by source path")
    parser.add_argument("--list-docs", action="store_true", help="List indexed sources and chunk counts")
    parser.add_argument("--clear", action="store_true", help="Clear the entire collection")
    parser.add_argument("--inspect-pdfs", action="store_true", help="Print PDF extraction stats (no indexing)")
    parser.add_argument("--inspect-dir", default="papers", help="Folder to inspect with --inspect-pdfs")

    args = parser.parse_args()

    # --inspect-pdfs does not need a populated store, but RAGSystem init is cheap.
    rag = RAGSystem(
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        embedding_host=args.embedding_host,
        llm_host=args.llm_host,
        llm_model=args.llm_model,
        persist_directory=args.persist_dir,
        vector_backend=args.vector_backend,
        pg_dsn=args.pg_dsn,
    )

    if args.inspect_pdfs:
        pdfs = [f for f in get_documents_from_folder(args.inspect_dir) if f.lower().endswith(".pdf")]
        if not pdfs:
            print(f"No PDFs found in '{args.inspect_dir}'.")
            return
        print(f"Inspecting {len(pdfs)} PDF(s) in '{args.inspect_dir}':\n")
        for pdf in pdfs:
            stats = rag.inspect_pdf(pdf)
            flag = "  ⚠️ LOOKS EMPTY" if stats["looks_empty"] else ""
            print(f"  {stats['filename']}: {stats['pages']} pages, "
                  f"{stats['chars']} chars (~{stats['chars_per_page']}/page){flag}")
        return

    if args.clear:
        rag.clear_collection()

    if args.add:
        rag.add_or_update_document(args.add)

    if args.remove:
        rag.remove_document(args.remove)

    if args.list_docs:
        docs = rag.list_documents()
        if not docs:
            print("No documents indexed.")
        else:
            print(f"{len(docs)} indexed source(s):")
            for d in docs:
                print(f"  - {d['source']} ({d['chunks']} chunks)")

    if args.build:
        folders = args.papers_dirs if args.papers_dirs else [args.papers_dir]
        build_index(rag, folders)

    if args.question:
        if rag.store.count() == 0:
            print("Collection is empty. Run with --build first.")
            return
        answer(rag, args.question, args)
        return

    if args.interactive:
        if rag.store.count() == 0:
            print("Collection is empty. Run with --build first.")
            return
        interactive_shell(rag, args)
        return

    if not any([args.build, args.add, args.remove, args.list_docs, args.clear]):
        parser.print_help()


if __name__ == "__main__":
    main()
