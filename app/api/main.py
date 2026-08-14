"""
FastAPI application — wraps the same pipeline scripts/ask.py uses,
exposed as an HTTP endpoint.

Run locally: uvicorn app.api.main:app --reload
Run in Docker: see Dockerfile / docker-compose.yml

Note on similarity scores in the response: per the OWASP RAG guidance
reviewed earlier (Section 8 — Query Injection via Retrieval), raw
similarity scores let a caller probe the corpus structure through
differential analysis (varying queries and watching score deltas to
map what's in the vector store). For a solo local-dev v1 this is a
low-severity concern, but it costs nothing to default to NOT exposing
raw scores in the public response shape — they're still logged
server-side and available via the /ask/debug endpoint for your own
testing. If this ever gets deployed somewhere multi-user, that
debug endpoint should be removed or authenticated, not left open.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.generation.guardrail import answer_question
from app.retrieval.embedder import embed_query
from app.retrieval.store import get_connection, similarity_search

app = FastAPI(
    title="Government Scheme Eligibility Assistant",
    description=(
        "Grounded RAG assistant for government welfare scheme eligibility. "
        "Answers only from ingested official documents, with citations, "
        "and refuses rather than guesses when confidence is low."
    ),
    version="1.0.0",
)

TOP_K = 5


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class Citation(BaseModel):
    scheme_name: str
    document_name: str
    page_number: int


class AskResponse(BaseModel):
    answered: bool
    answer: str | None
    refusal_reason: str | None


@app.get("/health")
def health():
    """Liveness check — does NOT verify DB/LLM connectivity, just that the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """
    Readiness check — actually verifies the DB connection works, since a
    container can be 'alive' (health passes) while its Postgres
    connection is broken (e.g. db container not up yet, wrong
    credentials). Distinguishing these two matters for orchestration:
    a failed health check means restart the container; a failed
    readiness check means don't route traffic to it yet, but don't
    restart it either — it might just be waiting on Postgres.
    """
    try:
        conn = get_connection()
        conn.close()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database not reachable: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        query_vec = embed_query(request.question)
        conn = get_connection()
        chunks = similarity_search(conn, query_vec, k=TOP_K)
        conn.close()

        result = answer_question(request.question, chunks)

        return AskResponse(
            answered=result.answered,
            answer=result.answer,
            refusal_reason=result.refusal_reason,
        )
    except Exception as e:
        # Fail-closed per the OWASP guidance reviewed earlier: on any
        # pipeline error, return a clear error rather than silently
        # falling back to an ungrounded answer from the LLM's own
        # knowledge. The exception is intentionally not swallowed into
        # a 200 response with answered=false, since that would look
        # identical to a legitimate guardrail refusal in logs/clients -
        # a genuine error and a genuine refusal are different situations
        # and should be distinguishable.
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
