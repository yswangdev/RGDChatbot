# RGD Chatbot — RAG System

A Retrieval-Augmented Generation (RAG) assistant for the **Rat Genome Database
(RGD)**. Its job is to guide users around the RGD website and tools. It uses
**Ollama** for embeddings and generation and **PostgreSQL + pgvector** for
vector storage (with ChromaDB available as a local fallback).

## Architecture

- **Embeddings / LLM**: Ollama (`mxbai-embed-large:latest` + `llama3.2`) at
  `http://grudge.rgd.mcw.edu:11434`.
- **Vector store**: pluggable ([`vectorstore.py`](vectorstore.py)) — `pgvector`
  (default, multi-process safe) or `chroma` (single-process fallback).
- **Chunking**: sentence- and markdown-header-aware
  ([`chunking.py`](chunking.py)); no more mid-sentence splits.
- **Prioritization**: [`corpus_config.py`](corpus_config.py) ranks website
  help (`RGDHelpMarkdown/`) above research papers (`papers/`).

## Repository layout

```
RAG.py            RAGSystem: embeddings, chunking, retrieval, cited generation
rag_cli.py        CLI: build/query/manage the index
vectorstore.py    VectorStore abstraction (pgvector default, chroma fallback)
chunking.py       Sentence/markdown-aware chunking
corpus_config.py  Document prioritization rules
schema.sql        Postgres + pgvector schema
docker-compose.yml  Local pgvector for development
RGDHelpMarkdown/  Website help docs (primary corpus)
papers/           Background research papers (secondary corpus)
eval/             Evaluation harness — see eval/METRICS.md
```

The **evaluation harness** ([`eval/`](eval)) scores retrieval and generation
with reference-based metrics + an LLM-as-judge and combines them into one
overall score; every metric is documented in [`eval/METRICS.md`](eval/METRICS.md).
It expects a help-desk export `qa_report.html` at the repo root — this file is
**git-ignored** because it contains personal names/emails, so supply your own.

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

## Evaluation (`eval/`)

A harness that scores the RAG system on a fixed set of real RGD help-desk
questions ([`qa_report.html`](qa_report.html)) using **two methods** — classic
**reference-based** metrics and **LLM-as-a-judge** — across both **retrieval**
and **generation**, plus a **robustness** stress test, combined into one
overall score.

```bash
pip install -r requirements.txt          # adds beautifulsoup4, sacrebleu, rouge-score, nltk, bert-score
python -m eval.run_eval --n 50 --pg-dsn "$PG_DSN"
# faster iteration:
python -m eval.run_eval --n 50 --limit 5 --no-bertscore
```

What it measures, all on the **same** auto-selected 50 answerable questions:

- **Reference-based retrieval**: precision@k, recall@k, hit-rate, MRR, NDCG@k,
  RAGAS context-recall — chunk relevance derived from the reference answers
  (lexical ROUGE-L + semantic embedding similarity).
- **Reference-based generation**: BLEU, ROUGE-1/2/L, METEOR, token-F1, BERTScore
  (RAG answer vs the cleaned human answer).
- **LLM-as-judge** (`qwen2.5:7b`): per-chunk relevance (0–3), context quality
  (1–5), and answer faithfulness / correctness / relevance / completeness (1–5).
- **Robustness**: each question re-asked with typos / case / noise / prompt
  injection; score = answer stability (embedding cosine vs the original answer).
- **Overall**: each family normalized to [0,1] and combined by a configurable
  weighted average ([`eval/scoring.py`](eval/scoring.py); override with `--weights`).

Outputs (under `data/eval/`, gitignored): `eval_report.html` and
`eval_results.csv` — a per-question table of `Q | reference A | RAG A | every
score | overall` plus an aggregate row — and `summary.json`.

Useful flags: `--n` (set size), `--limit` (debug subset), `--k` (top-k),
`--no-bertscore` (skip the heavy torch download), `--no-judge`, `--no-stress`,
`--judge-model`, `--rebuild-set`, `--vector-backend chroma`.
