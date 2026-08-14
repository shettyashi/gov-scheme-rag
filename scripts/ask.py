"""
Phase 4 entry point — full question-answering pipeline.

Usage:
    python scripts/ask.py "Am I eligible for PM-KISAN if I own 3 acres?"

Requires GROQ_API_KEY set as an environment variable.
"""

import os
import sys

from app.generation.guardrail import answer_question
from app.retrieval.embedder import embed_query
from app.retrieval.store import get_connection, similarity_search

TOP_K = 5


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question here"')
        return

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set. Get a free key at "
              "https://console.groq.com/keys and set it, e.g.:\n"
              "  set GROQ_API_KEY=your_key_here   (Windows cmd)")
        return

    question = sys.argv[1]
    print(f"Question: {question}\n")

    query_vec = embed_query(question)
    conn = get_connection()
    chunks = similarity_search(conn, query_vec, k=TOP_K)
    conn.close()

    result = answer_question(question, chunks)

    if result.answered:
        print("ANSWERED")
        print(f"(top similarity: {result.top_similarity:.3f}, self-check: passed)\n")
        print(result.answer)
    else:
        print("REFUSED")
        print(f"(top similarity: {result.top_similarity:.3f}, "
              f"self-check: {result.self_check_passed})\n")
        print(f"Reason: {result.refusal_reason}")


if __name__ == "__main__":
    main()
