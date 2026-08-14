"""
Phase 3 entry point — manual retrieval testing.

Usage:
    python scripts/query.py "Am I eligible for PM-KISAN if I own 3 acres?"

This does NOT generate an answer yet (that's Phase 4). It only shows
you what the retriever actually pulls back and how confident it is,
so you can sanity-check retrieval quality in isolation before the LLM
generation step makes it harder to tell whether a wrong answer came
from bad retrieval or bad generation.

What to look for when eyeballing results:
- Do the top chunks actually relate to the question's scheme AND topic
  (not just the right scheme, wrong topic within it)?
- Is there a clear score gap between the top result and the rest, or
  does similarity decay smoothly (meaning nothing is confidently "the"
  answer)? A smooth decay with no gap is a signal that this query will
  be a good test case for tuning the Phase 4 refusal threshold.
- For a question with NO real answer in your corpus (e.g. asking about
  a scheme you didn't ingest), does the top score drop noticeably
  lower than for an answerable question? If confident-looking scores
  show up even for nonsense queries, cosine similarity alone won't be
  a reliable refusal signal and Phase 4 will need the extra LLM
  self-check, not just a threshold.
"""

import sys

from app.retrieval.embedder import embed_query
from app.retrieval.store import get_connection, similarity_search


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/query.py "your question here"')
        return

    question = sys.argv[1]
    print(f"Query: {question}\n")

    query_vec = embed_query(question)
    conn = get_connection()
    results = similarity_search(conn, query_vec, k=5)
    conn.close()

    if not results:
        print("No results — is the database empty? Run embed_and_store.py first.")
        return

    for i, r in enumerate(results, start=1):
        ocr_tag = " [OCR]" if r.used_ocr else ""
        print(f"#{i}  similarity={r.similarity:.4f}  "
              f"{r.scheme_name} / {r.document_name} p{r.page_number}{ocr_tag}")
        print(f"    {r.chunk_text[:250].strip()}...")
        print()


if __name__ == "__main__":
    main()
