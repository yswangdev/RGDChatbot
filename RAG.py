"""
RAG (Retrieval-Augmented Generation) System using Ollama.

Embeddings and generation are served by Ollama; vector storage is handled by a
pluggable backend (PostgreSQL + pgvector by default, ChromaDB as a fallback).
Supports document loading, sentence-aware chunking, incremental indexing, and
source-cited retrieval-augmented generation.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import ollama
import pdfplumber
import PyPDF2

from chunking import chunk_document
from corpus_config import priority_for_source
from vectorstore import build_vector_store


class RAGSystem:
    """RAG system: Ollama embeddings/LLM + a pluggable vector store."""

    def __init__(
        self,
        collection_name: str = "rgd_chunks",
        embedding_model: str = "mxbai-embed-large:latest",
        llm_model: str = "llama3.2",
        persist_directory: str = "./chroma_db",
        embedding_host: str = "http://grudge.rgd.mcw.edu:11434",
        llm_host: str = "http://grudge.rgd.mcw.edu:11434",
        vector_backend: str = "pgvector",
        pg_dsn: Optional[str] = None,
    ):
        """Initialize the RAG system.

        Args:
            collection_name: Vector store collection / table name.
            embedding_model: Ollama embedding model.
            llm_model: Ollama generation model.
            persist_directory: Directory for ChromaDB data + links dictionary.
            embedding_host: Ollama server URL for embeddings.
            llm_host: Ollama server URL for chat/generation.
            vector_backend: "pgvector" (default) or "chroma".
            pg_dsn: PostgreSQL connection string (required for pgvector).
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_directory = persist_directory
        self.embedding_host = embedding_host
        self.llm_host = llm_host
        self.vector_backend = vector_backend
        self.embedding_client = ollama.Client(host=embedding_host)
        self.llm_client = ollama.Client(host=llm_host)

        self.store = build_vector_store(
            vector_backend,
            pg_dsn=pg_dsn,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        # Dictionary of URLs -> {annotation, context, source_file}.
        os.makedirs(persist_directory, exist_ok=True)
        self.links_file = os.path.join(persist_directory, "links_dict.json")
        self.links_dict: Dict[str, Dict] = {}
        self._load_links()

        print("✓ RAG System initialized")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - LLM model: {llm_model}")
        print(f"  - Embedding host: {embedding_host}")
        print(f"  - Vector backend: {vector_backend}")
        print(f"  - Collection: {collection_name}")
        print(f"  - Indexed chunks: {self.store.count()}")
        print(f"  - Links dictionary: {len(self.links_dict)} links loaded")

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def _get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text using Ollama."""
        try:
            response = self.embedding_client.embeddings(model=self.embedding_model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Chunk, embed, and store a list of documents.

        Each document's metadata must include a ``source``; ``file_type`` and
        ``priority`` are used when present. Chunk ids are derived from the
        source and a per-source running index so re-indexing a source upserts
        cleanly.
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)

        all_ids, all_embeddings, all_chunks, all_metadatas = [], [], [], []
        per_source_counter: Dict[str, int] = {}

        for doc, meta in zip(documents, metadatas):
            meta = dict(meta or {})
            source = meta.get("source", "unknown")
            meta.setdefault("priority", priority_for_source(source))
            chunks = chunk_document(
                doc,
                file_type=meta.get("file_type", "text"),
                source=source,
                base_metadata=meta,
            )
            for chunk in chunks:
                idx = per_source_counter.get(source, 0)
                per_source_counter[source] = idx + 1
                all_ids.append(f"{source}::{idx}")
                all_chunks.append(chunk.content)
                all_metadatas.append(chunk.metadata)

        if not all_chunks:
            print("No quality chunks produced.")
            return

        print(f"Processing {len(all_chunks)} chunks...")
        for i, chunk in enumerate(all_chunks):
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(all_chunks)} chunks...")
            all_embeddings.append(self._get_embedding(chunk))

        self.store.add(all_ids, all_embeddings, all_chunks, all_metadatas)
        print(f"✓ Added {len(all_chunks)} chunks to vector store")

    def add_documents_from_files(self, file_paths: List[str]):
        """Load documents from files (PDF/text/markdown) and index them."""
        documents, metadatas = [], []
        for file_path in file_paths:
            loaded = self._load_file(file_path)
            if loaded is not None:
                content, meta = loaded
                documents.append(content)
                metadatas.append(meta)

        if documents:
            print(f"\nProcessing {len(documents)} document(s)...")
            self.add_documents(documents, metadatas)
            if self.links_dict:
                self._save_links()
                print(f"  ✓ Saved {len(self.links_dict)} links to dictionary")

    def _load_file(self, file_path: str):
        """Read one file, returning (content, metadata) or None on failure/empty."""
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            return None

        try:
            if path.suffix.lower() == ".pdf":
                print(f"Extracting text from PDF: {path.name}...")
                content, page_count = self._extract_text_from_pdf(str(path))
                if not content.strip():
                    print(f"Warning: No text extracted from {file_path} (scanned/image PDF?)")
                    return None
                file_type, extra = "pdf", {"pages": page_count}
                print(f"  ✓ Extracted {len(content)} chars across {page_count} page(s) from {path.name}")
            else:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                file_type = "markdown" if path.suffix.lower() == ".md" else "text"
                extra = {}

            self._collect_links(content, path.name)
            meta = {
                "source": str(path),
                "filename": path.name,
                "file_type": file_type,
                "content_hash": self._hash_text(content),
                "priority": priority_for_source(str(path)),
                **extra,
            }
            return content, meta
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Incremental document management
    # ------------------------------------------------------------------ #
    def add_or_update_document(self, file_path: str) -> str:
        """Index a single document, skipping it if unchanged since last index.

        Returns one of "skipped", "added", or "updated".
        """
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            return "missing"

        new_hash = self._hash_file(path)
        existing = self.store.source_hashes(str(path))
        if existing and new_hash in existing:
            print(f"= Unchanged, skipping: {path.name}")
            return "skipped"

        status = "updated" if existing else "added"
        if existing:
            removed = self.store.delete({"source": str(path)})
            print(f"↻ Re-indexing {path.name} (removed {removed} stale chunk(s))")
        self.add_documents_from_files([str(path)])
        return status

    def remove_document(self, source: str) -> int:
        """Remove all chunks for a source. Returns number of chunks deleted."""
        deleted = self.store.delete({"source": source})
        print(f"✓ Removed {deleted} chunk(s) for source: {source}")
        return deleted

    def list_documents(self) -> List[Dict]:
        """Return indexed sources with chunk counts."""
        return self.store.list_sources()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    # ------------------------------------------------------------------ #
    # PDF extraction
    # ------------------------------------------------------------------ #
    def _extract_text_from_pdf(self, file_path: str):
        """Extract text from a PDF, returning ``(text, page_count)``.

        Uses pdfplumber first, falling back to PyPDF2. De-hyphenates words
        broken across line endings to avoid corrupting tokens.
        """
        pages_text: List[str] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
        except Exception as e:
            print(f"Warning: pdfplumber failed for {file_path}: {e}, trying PyPDF2...")
            pages_text = []

        if not pages_text:
            try:
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(page_text)
            except Exception as e:
                print(f"Error extracting text from PDF {file_path}: {e}")
                raise

        text = "\n\n".join(pages_text)
        # Join words split by a hyphen at a line break: "geno-\nme" -> "genome".
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return text, len(pages_text)

    def inspect_pdf(self, file_path: str) -> Dict:
        """Return extraction statistics for a single PDF (no indexing)."""
        path = Path(file_path)
        text, page_count = self._extract_text_from_pdf(str(path))
        stripped = text.strip()
        return {
            "filename": path.name,
            "pages": page_count,
            "chars": len(stripped),
            "chars_per_page": (len(stripped) // page_count) if page_count else 0,
            "looks_empty": len(stripped) < 100,
        }

    # ------------------------------------------------------------------ #
    # Links dictionary
    # ------------------------------------------------------------------ #
    def _collect_links(self, content: str, source_file: str):
        for url_info in self._extract_urls_from_text(content):
            url = url_info["url"]
            if url not in self.links_dict:
                self.links_dict[url] = {
                    "annotation": url_info["annotation"],
                    "context": url_info["context"],
                    "source_file": source_file,
                }

    def _extract_urls_from_text(self, text: str, context_window: int = 200) -> List[Dict]:
        """Extract URLs from text with surrounding context for annotation."""
        urls = []
        url_pattern = (
            r'(https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
            r'|rgd\.mcw\.edu[^\s<>"{}|\\^`\[\]]*|doi\.org/[^\s<>"{}|\\^`\[\]]+)'
        )
        for match in re.finditer(url_pattern, text, re.IGNORECASE):
            url = match.group(0)
            start_pos, end_pos = match.start(), match.end()
            context_start = max(0, start_pos - context_window)
            context_end = min(len(text), end_pos + context_window)
            context = text[context_start:context_end].strip()

            annotation = ""
            context_lower = context.lower()
            if any(p in context_lower for p in ["available at", "visit", "access", "see", "download", "find"]):
                for sentence in re.split(r"[.!?]\s+", context):
                    if url.lower() in sentence.lower():
                        annotation = sentence.strip()
                        break
            if not annotation:
                annotation = context[:150] + "..." if len(context) > 150 else context

            urls.append({"url": url, "context": context, "annotation": annotation, "position": start_pos})
        return urls

    def _load_links(self):
        if os.path.exists(self.links_file):
            try:
                with open(self.links_file, "r", encoding="utf-8") as f:
                    self.links_dict = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load links dictionary: {e}")
                self.links_dict = {}

    def _save_links(self):
        try:
            os.makedirs(os.path.dirname(self.links_file), exist_ok=True)
            with open(self.links_file, "w", encoding="utf-8") as f:
                json.dump(self.links_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save links dictionary: {e}")

    def _find_relevant_links(self, query: str, max_links: int = 3) -> List[Dict]:
        """Find links whose annotation/context overlaps the query keywords."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        relevant = []
        for url, info in self.links_dict.items():
            annotation = info.get("annotation", "").lower()
            context = info.get("context", "").lower()
            score = len(query_words & set(annotation.split())) * 2
            score += len(query_words & set(context.split()))
            if any(w in annotation or w in context for w in ["download", "access", "tool", "database", "portal"]):
                if any(w in query_lower for w in ["download", "access", "tool", "database", "portal", "link", "url"]):
                    score += 3
            if score > 0:
                relevant.append({"url": url, "annotation": info.get("annotation", ""), "score": score})
        relevant.sort(key=lambda x: x["score"], reverse=True)
        return relevant[:max_links]

    # ------------------------------------------------------------------ #
    # Retrieval & generation
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, n_results: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self._get_embedding(query)
        return self.store.query(
            query_embedding,
            n_results=n_results,
            min_similarity=min_similarity,
        )

    DEFAULT_SYSTEM_PROMPT = """You are the Rat Genome Database (RGD) assistant. Your job is to help \
users navigate and use the RGD website and its tools, based ONLY on the provided context.

Expected questions are about how to use RGD: site/gene/ontology search, disease portals, \
JBrowse and other genome browsers, ontology browsers, PhenoMiner, gene/QTL/strain/marker \
report pages, evidence codes, and nomenclature.

INSTRUCTIONS:
1. Answer using ONLY the information in the provided context. Do not use outside knowledge.
2. Be clear and concise. For "how do I..." questions, give short numbered steps.
3. Cite the sources you used inline as [filename] (and the section when given). When the \
context includes a relevant RGD link, include it so the user can click through.
4. If the context does not contain the answer, say: "I cannot find information about this in \
the available RGD documentation." Do not guess.
"""

    def generate(
        self,
        query: str,
        n_results: int = 5,
        system_prompt: Optional[str] = None,
        min_similarity: float = 0.3,
        include_sources: bool = True,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> str:
        """Retrieve context and generate a source-cited answer."""
        retrieved_docs = self.retrieve(query, n_results=n_results, min_similarity=min_similarity)
        if not retrieved_docs:
            return "I cannot find information about this in the available RGD documentation."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.get("metadata", {})
            filename = meta.get("filename", "Unknown")
            section = meta.get("section")
            label = f"{filename} — {section}" if section else filename
            context_parts.append(f"[Source {i}: {label}]\n{doc['document']}")
        context = "\n\n---\n\n".join(context_parts)

        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        prompt = f"""Use the following RGD documentation context to answer the question.

CONTEXT:
{context}

QUESTION: {query}

Answer (cite sources inline as [filename]):"""

        try:
            options: Dict[str, object] = {"temperature": temperature}
            if seed is not None:
                options["seed"] = seed
            response = self.llm_client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options=options,
            )
            answer = response["message"]["content"].strip()

            no_info_phrases = ["cannot find", "don't have", "not available", "not found", "no information", "doesn't contain"]
            if any(p in answer.lower() for p in no_info_phrases):
                if retrieved_docs and retrieved_docs[0].get("similarity", 0) < 0.4:
                    return "I cannot find information about this in the available RGD documentation."

            if include_sources:
                answer += "\n\n" + self._format_sources(query, retrieved_docs)
            return answer
        except Exception as e:
            return f"Error generating response: {e}"

    def _format_sources(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Render a compact Sources block from retrieved metadata + matched links."""
        seen = set()
        lines = []
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            filename = meta.get("filename", "Unknown")
            section = meta.get("section")
            page = meta.get("page")
            label = filename
            if section:
                label += f" — {section}"
            elif page:
                label += f" — p. {page}"
            if label not in seen:
                seen.add(label)
                lines.append(f"  - {label}")

        links = self._find_relevant_links(query)
        for link in links:
            entry = f"  - {link['url']}"
            if entry not in seen:
                seen.add(entry)
                lines.append(entry)

        if not lines:
            return ""
        return "Sources:\n" + "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    def clear_collection(self):
        """Clear all chunks from the vector store."""
        self.store.clear()
        print("✓ Collection cleared")
