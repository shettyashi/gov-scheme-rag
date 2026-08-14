"""
Phase 1 entry point.

Usage:
    python scripts/ingest.py

Reads data/raw/<scheme_name>/*.pdf, extracts + chunks them, writes
data/processed/chunks.jsonl. Embedding happens in Phase 2 — this script
deliberately stops before touching Postgres, so you can inspect chunk
quality before paying for/waiting on embeddings.
"""

import json
from dataclasses import asdict
from pathlib import Path

from app.ingestion.loader import load_raw_pages
from app.ingestion.chunker import chunk_pages

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"No data in {RAW_DIR}. Download official scheme PDFs into "
              f"data/raw/<scheme_name>/<document>.pdf first — see "
              f"app/ingestion/loader.py docstring for the expected layout.")
        return

    pages = load_raw_pages(RAW_DIR)
    print(f"Loaded {len(pages)} raw pages.")

    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunks "
          f"(avg {sum(len(c.chunk_text) for c in chunks) / max(len(chunks),1):.0f} chars/chunk).")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
