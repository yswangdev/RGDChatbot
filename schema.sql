-- Schema for the RGD chatbot vector store (PostgreSQL + pgvector).
-- The application also creates these objects automatically on startup
-- (see PgVectorStore._ensure_schema); this file documents the schema and can
-- be applied manually, e.g.:
--   psql "$PG_DSN" -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- vector(1024) matches the mxbai-embed-large embedding dimension.
CREATE TABLE IF NOT EXISTS rgd_chunks (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    content      TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding    vector(1024),
    priority     INT NOT NULL DEFAULT 1,
    content_hash TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rgd_chunks_source_idx ON rgd_chunks (source);

-- HNSW index for fast cosine-similarity search.
CREATE INDEX IF NOT EXISTS rgd_chunks_embedding_idx
    ON rgd_chunks USING hnsw (embedding vector_cosine_ops);
