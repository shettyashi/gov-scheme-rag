"""
Postgres/pgvector storage layer.

Connection config comes from environment variables (12-factor style),
with local-dev defaults so `python scripts/embed_and_store.py` works
against docker-compose out of the box.
"""

import os
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from pgvector.utils import to_db

DB_CONFIG = dict(
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
    dbname=os.environ.get("PGDATABASE", "gov_scheme_rag"),
    user=os.environ.get("PGUSER", "rag"),
    password=os.environ.get("PGPASSWORD", "rag"),
)


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(**DB_CONFIG, autocommit=True)
    register_vector(conn)
    return conn


def insert_chunks_batch(conn: psycopg.Connection, rows: list[dict]) -> None:
    """
    rows: each dict must have scheme_name, document_name, source_url,
    page_number, chunk_text, chunk_index, used_ocr, embedding (list[float]).

    Uses parameterized executemany rather than COPY: COPY's text-format
    protocol serializes Python lists as Postgres array literals
    ("{1,2,3}"), which pgvector's column type rejects — it requires its
    own bracket syntax ("[1,2,3]"). register_vector's adapter handles
    that conversion correctly for parameterized queries but not for
    COPY, so COPY silently produces a value Postgres can't parse as a
    vector. executemany is slower for very large batches, but at v1's
    scale (hundreds to low thousands of chunks) the difference is
    immaterial — correctness here matters more than COPY's speed edge.
    """
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO scheme_chunks
                (scheme_name, document_name, source_url, page_number,
                 chunk_text, chunk_index, used_ocr, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    row["scheme_name"],
                    row["document_name"],
                    row["source_url"],
                    row["page_number"],
                    # OCR output (and occasionally raw PDF text extraction)
                    # can contain NUL bytes, which Postgres text columns
                    # reject outright. Strip them here rather than at
                    # extraction time, so the raw chunks.jsonl file still
                    # reflects exactly what the loader/chunker produced.
                    row["chunk_text"].replace("\x00", ""),
                    row["chunk_index"],
                    row["used_ocr"],
                    row["embedding"],
                )
                for row in rows
            ],
        )


@dataclass
class RetrievedChunk:
    id: int
    scheme_name: str
    document_name: str
    source_url: str
    page_number: int
    chunk_text: str
    used_ocr: bool
    similarity: float  # cosine similarity, 1.0 = identical, higher = closer


def similarity_search(conn: psycopg.Connection, query_embedding: list[float], k: int = 5) -> list["RetrievedChunk"]:
    """
    Top-k nearest chunks by cosine similarity.

    pgvector's `<=>` operator returns cosine DISTANCE (0 = identical,
    2 = opposite), not similarity — we convert to similarity
    (1 - distance) here so callers work with the more intuitive
    "higher = better match" convention used everywhere else in the
    pipeline (e.g. the confidence guardrail threshold in Phase 4).
    """
    with conn.cursor() as cur:
        qvec = to_db(query_embedding)
        cur.execute(
            """
            SELECT id, scheme_name, document_name, source_url, page_number,
                   chunk_text, used_ocr, 1 - (embedding <=> %s::vector) AS similarity
            FROM scheme_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, k),
        )
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            id=r[0], scheme_name=r[1], document_name=r[2], source_url=r[3],
            page_number=r[4], chunk_text=r[5], used_ocr=r[6], similarity=float(r[7]),
        )
        for r in rows
    ]
