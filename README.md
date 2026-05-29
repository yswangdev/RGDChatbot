# RGD Chatbot — RAG System

A Retrieval-Augmented Generation (RAG) assistant for the **Rat Genome Database
(RGD)**. Its job is to guide users around the RGD website and tools. It uses
**Ollama** for embeddings and generation and **PostgreSQL + pgvector** for
vector storage (with ChromaDB available as a local fallback).

This is **Phase 1** (the retrieval foundation). A web app with streaming,
per-session conversation memory, and an upload/delete UI is planned for Phase 2.

## Architecture

- **Embeddings / LLM**: Ollama (`mxbai-embed-large:latest` + `llama3.2`) at
  `http://grudge.rgd.mcw.edu:11434`.
- **Vector store**: pluggable ([`vectorstore.py`](vectorstore.py)) — `pgvector`
  (default, multi-process safe) or `chroma` (single-process fallback).
- **Chunking**: sentence- and markdown-header-aware
  ([`chunking.py`](chunking.py)); no more mid-sentence splits.
- **Prioritization**: [`corpus_config.py`](corpus_config.py) ranks website
  help (`RGDHelpMarkdown/`) above research papers (`papers/`).

## Setup

1. **Virtual environment + dependencies**
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start PostgreSQL + pgvector** (local dev via Docker)
   ```bash
   docker compose up -d
   export PG_DSN="postgresql://rgd:rgd@localhost:5432/rgd"
   ```
   The schema ([`schema.sql`](schema.sql)) is applied automatically — both by
   the container on first init and by the app on startup.

   To use ChromaDB instead (single process only), pass `--vector-backend chroma`.

3. **Inspect PDF extraction** (verify the corpus parses before indexing)
   ```bash
   python rag_cli.py --inspect-pdfs
   ```

4. **Build the index** (incremental — re-runs skip unchanged files)
   ```bash
   python rag_cli.py --build --papers-dirs RGDHelpMarkdown papers
   ```

5. **Ask questions**
   ```bash
   python rag_cli.py --question "How do I search for a gene in RGD?"
   python rag_cli.py --interactive
   ```

## Document management

```bash
python rag_cli.py --add papers/new_paper.pdf        # add or update one document
python rag_cli.py --remove RGDHelpMarkdown/glossary.md   # remove by source path
python rag_cli.py --list-docs                       # list sources + chunk counts
python rag_cli.py --clear                           # wipe the collection
```

Re-indexing is incremental: each source's content is hashed, so `--build`
skips files that haven't changed and only re-embeds those that have.

## Responses & citations

Answers are grounded only in the indexed RGD documentation. The system prompt
instructs the model to give numbered steps for "how do I..." questions and to
cite sources inline as `[filename]`. A `Sources:` block (filenames, sections /
page numbers, and any relevant RGD link) is appended to each answer.

## Key CLI arguments

```
--build              Incrementally (re)build the index
--papers-dir DIR     Single documents folder (default: RGDHelpMarkdown)
--papers-dirs ...    Multiple folders to index
--vector-backend     pgvector (default) | chroma
--pg-dsn DSN         PostgreSQL DSN (or set PG_DSN env)
--collection NAME    Collection / table name (default: rgd_chunks)
--embedding-model    Ollama embedding model (default: mxbai-embed-large:latest)
--llm-model          LLM model (default: llama3.2)
--question Q         Ask a single question
--interactive        Interactive Q&A shell
--n-results N        Chunks to retrieve (default: 5)
--min-similarity F   Minimum similarity threshold (default: 0.3)
--show-context       Print retrieved chunks
--add PATH           Add/update one document
--remove SOURCE      Remove a document by source path
--list-docs          List indexed sources
--clear              Clear the collection
--inspect-pdfs       Print PDF extraction stats (no indexing)
```
