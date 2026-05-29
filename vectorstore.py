"""
Vector store abstraction with PostgreSQL/pgvector and ChromaDB backends.

ChromaDB's persistent client only supports a single process, which is a
non-starter for a production web service. The primary backend is therefore
PostgreSQL + pgvector, which supports concurrent connections and SQL-based
filtering/deletion (needed for incremental document management). The ChromaDB
backend is kept behind the same interface for local fallback and comparison.
"""

import json
from typing import Dict, List, Optional, Protocol, Sequence

from corpus_config import PRIORITY_BOOST

# Embedding dimensionality. mxbai-embed-large produces 1024-dim vectors; change
# this (and re-index) if you swap the embedding model.
EMBEDDING_DIM = 1024


class VectorStore(Protocol):
    """Storage-agnostic interface used by the RAG system."""

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict],
    ) -> None: ...

    def query(
        self,
        embedding: Sequence[float],
        n_results: int = 5,
        min_similarity: float = 0.0,
        where: Optional[Dict] = None,
    ) -> List[Dict]: ...

    def delete(self, where: Dict) -> int: ...

    def count(self) -> int: ...

    def list_sources(self) -> List[Dict]: ...

    def source_hashes(self, source: str) -> List[str]: ...

    def clear(self) -> None: ...


class PgVectorStore:
    """PostgreSQL + pgvector backed store."""

    def __init__(
        self,
        dsn: str,
        table: str = "rgd_chunks",
        dim: int = EMBEDDING_DIM,
    ):
        import psycopg
        from pgvector.psycopg import register_vector

        self._psycopg = psycopg
        self.table = table
        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._ensure_schema()
        register_vector(self.conn)

    def _ensure_schema(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id           TEXT PRIMARY KEY,
                source       TEXT NOT NULL,
                content      TEXT NOT NULL,
                metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding    vector({self.dim}),
                priority     INT NOT NULL DEFAULT 1,
                content_hash TEXT,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_source_idx ON {self.table} (source)"
        )
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
            ON {self.table} USING hnsw (embedding vector_cosine_ops)
            """
        )

    def add(self, ids, embeddings, documents, metadatas) -> None:
        import numpy as np

        rows = []
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            meta = dict(meta or {})
            rows.append(
                (
                    id_,
                    meta.get("source", ""),
                    doc,
                    json.dumps(meta),
                    np.array(emb, dtype=np.float32),
                    int(meta.get("priority", 1)),
                    meta.get("content_hash"),
                )
            )
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table}
                    (id, source, content, metadata, embedding, priority, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    source = EXCLUDED.source,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    priority = EXCLUDED.priority,
                    content_hash = EXCLUDED.content_hash
                """,
                rows,
            )

    def query(self, embedding, n_results=5, min_similarity=0.0, where=None) -> List[Dict]:
        import numpy as np

        # Fetch a wider candidate set, then re-rank with the priority boost and
        # apply the similarity floor on the raw cosine similarity.
        fetch_n = max(n_results * 4, 20)
        sql = f"""
            SELECT content, metadata, priority,
                   1 - (embedding <=> %s) AS similarity
            FROM {self.table}
        """
        params: List[object] = [np.array(embedding, dtype=np.float32)]
        clause, where_params = self._where_clause(where)
        if clause:
            sql += f" WHERE {clause}"
            params.extend(where_params)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params.append(np.array(embedding, dtype=np.float32))
        params.append(fetch_n)

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = []
        for content, metadata, priority, similarity in rows:
            if similarity < min_similarity:
                continue
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata)
            results.append(
                {
                    "document": content,
                    "metadata": meta,
                    "similarity": float(similarity),
                    "score": float(similarity) + PRIORITY_BOOST * int(priority),
                }
            )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:n_results]

    def delete(self, where: Dict) -> int:
        clause, params = self._where_clause(where)
        if not clause:
            raise ValueError("delete requires a non-empty `where` filter")
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE {clause}", params)
            return cur.rowcount

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return cur.fetchone()[0]

    def list_sources(self) -> List[Dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT source, count(*), max(priority), max(content_hash)
                FROM {self.table} GROUP BY source ORDER BY source
                """
            )
            return [
                {"source": s, "chunks": c, "priority": p, "content_hash": h}
                for s, c, p, h in cur.fetchall()
            ]

    def source_hashes(self, source: str) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT content_hash FROM {self.table} WHERE source = %s",
                [source],
            )
            return [r[0] for r in cur.fetchall() if r[0] is not None]

    def clear(self) -> None:
        self.conn.execute(f"TRUNCATE TABLE {self.table}")

    @staticmethod
    def _where_clause(where: Optional[Dict]):
        """Translate a simple ``{column/key: value}`` filter into SQL."""
        if not where:
            return "", []
        columns = {"source", "content_hash", "id", "priority"}
        clauses, params = [], []
        for key, value in where.items():
            if key in columns:
                clauses.append(f"{key} = %s")
            else:
                clauses.append("metadata->>%s = %s")
                params.append(key)
            params.append(value)
        return " AND ".join(clauses), params


class ChromaVectorStore:
    """ChromaDB-backed store, kept for local fallback (single-process only)."""

    def __init__(self, persist_directory: str, collection_name: str):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, documents, metadatas) -> None:
        # Chroma metadata values must be scalars; serialize anything else.
        flat = []
        for meta in metadatas:
            flat.append(
                {
                    k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v))
                    for k, v in (meta or {}).items()
                }
            )
        self.collection.add(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=flat,
        )

    def query(self, embedding, n_results=5, min_similarity=0.0, where=None) -> List[Dict]:
        fetch_n = max(n_results * 4, 20)
        results = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=fetch_n,
            where=where or None,
        )
        out = []
        docs = results.get("documents") or [[]]
        if docs and docs[0]:
            for i, doc in enumerate(docs[0]):
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                similarity = 1.0 - distance
                if similarity < min_similarity:
                    continue
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                priority = int(meta.get("priority", 1)) if meta else 1
                out.append(
                    {
                        "document": doc,
                        "metadata": meta,
                        "similarity": similarity,
                        "score": similarity + PRIORITY_BOOST * priority,
                    }
                )
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:n_results]

    def delete(self, where: Dict) -> int:
        before = self.collection.count()
        self.collection.delete(where=where)
        return before - self.collection.count()

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> List[Dict]:
        data = self.collection.get(include=["metadatas"])
        counts: Dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            src = (meta or {}).get("source", "")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": s, "chunks": c} for s, c in sorted(counts.items())]

    def source_hashes(self, source: str) -> List[str]:
        data = self.collection.get(where={"source": source}, include=["metadatas"])
        hashes = {(m or {}).get("content_hash") for m in data.get("metadatas") or []}
        return [h for h in hashes if h]

    def clear(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


def build_vector_store(
    backend: str,
    *,
    pg_dsn: Optional[str] = None,
    persist_directory: str = "./chroma_db",
    collection_name: str = "rgd_chunks",
) -> VectorStore:
    """Factory selecting the configured backend."""
    if backend == "pgvector":
        if not pg_dsn:
            raise ValueError("pgvector backend requires a connection string (--pg-dsn / PG_DSN)")
        return PgVectorStore(dsn=pg_dsn, table=collection_name)
    if backend == "chroma":
        return ChromaVectorStore(persist_directory=persist_directory, collection_name=collection_name)
    raise ValueError(f"Unknown vector backend: {backend}")
