CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS scheme_chunks (
    id              SERIAL PRIMARY KEY,
    scheme_name     TEXT NOT NULL,
    document_name   TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    page_number     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    used_ocr        BOOLEAN NOT NULL DEFAULT FALSE,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding       vector(768)   -- dim matches multilingual-e5-base; change if you swap models
);

-- HNSW index for cosine similarity search
CREATE INDEX IF NOT EXISTS scheme_chunks_embedding_idx
    ON scheme_chunks
    USING hnsw (embedding vector_cosine_ops);

-- Useful for filtering retrieval by scheme before/alongside vector search
CREATE INDEX IF NOT EXISTS scheme_chunks_scheme_name_idx
    ON scheme_chunks (scheme_name);
