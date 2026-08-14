"""
Phase 2 entry point.

Usage:
    python scripts/embed_and_store.py

Reads data/processed/chunks.jsonl (produced by scripts/ingest.py),
embeds each chunk with multilingual-e5-base, and writes them into
Postgres/pgvector in batches.

Batching rationale: embedding 500+ chunks one at a time round-trips to
the model for each call, which is slow even locally and needlessly
slow if the model is ever swapped for an API-based one. Batch encoding
lets sentence-transformers parallelize internally.
"""

import json
from pathlib import Path

from app.retrieval.embedder import embed_passages
from app.retrieval.store import get_connection, insert_chunks_batch

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
BATCH_SIZE = 32


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    if not CHUNKS_PATH.exists():
        print(f"No {CHUNKS_PATH} found. Run scripts/ingest.py first.")
        return

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    conn = get_connection()
    total_inserted = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]
        embeddings = embed_passages(texts)

        rows = [
            {**chunk, "embedding": embedding}
            for chunk, embedding in zip(batch, embeddings)
        ]
        insert_chunks_batch(conn, rows)
        total_inserted += len(rows)
        print(f"  Embedded + inserted {total_inserted}/{len(chunks)}")

    conn.close()
    print(f"Done. {total_inserted} chunks embedded and stored.")


if __name__ == "__main__":
    main()
