# Grounded RAG Assistant for Government Scheme Eligibility

It is a retrieval-augmented pipeline that answers government welfare scheme eligibility
questions using only official scheme documents — with page-level citations, and an
explicit refusal when it isn't confident, rather than a guess.

## Problem

General-purpose LLM chatbots hallucinate on the specifics of evolving government
welfare-scheme rules. Citizens have no reliable single place to check eligibility
grounded in official documentation. This project builds a pipeline that only answers
from ingested official PDFs, and measurably refuses rather than guesses when the
retrieved content doesn't actually support an answer.

## Results (not targets — measured, on an 30-question held-out eval set)

| Metric | Target | Measured |
|---|---|---|
| Retrieval hit-rate@5 | >85% | **90%** (27/30 scored questions) |
| Hallucination rate | <5% | **3.3%** (1/30 answered questions) |
| Refusal accuracy | — | **6.67%** (2 false answers, 0 false refusals) |


## Schemes covered (v1)

PM-KISAN, PMAY (Pradhan Mantri Awas Yojana), PM-MUDRA — 3 schemes, 9 official
documents, 576 chunks. PM-JAY (Ayushman Bharat) was considered and deliberately
dropped — not enough verifiable official source documents were available to ingest it
responsibly, and the eval set uses PM-JAY questions specifically as refusal test cases
(the system correctly identifies it has no data on this scheme rather than guessing).

## Architecture

```
Official Scheme PDFs (manually collected)
        │
   PDF Loader (pdfplumber, per-page extraction)
        │
   Low-confidence page? ──yes──> Tesseract OCR (eng+hin) ──> still empty? ──> excluded, logged
        │ no                                                        │
        └────────────────────────┬─────────────────────────────────┘
                                  │
                        Chunker (recursive splitter, page-tagged,
                         cross-page-boundary aware, ~800 chars/20% overlap)
                                  │
                    Embedding (multilingual-e5-base, query/passage-prefixed)
                                  │
                    PostgreSQL + pgvector (HNSW, cosine similarity)
                                  │
                    Retriever (top-k similarity search)
                                  │
        Similarity < 0.78? ──yes──> REFUSE (no LLM call spent)
                    │ no
        LLM self-check: "does retrieved content actually answer this?"
                    │
              NO ──> REFUSE, with reason
                    │ YES
        Grounded generation (retrieved content delimited as DATA, not
         instructions — see Security section) with per-claim citations
                    │
        Faithfulness check (separate LLM-judge call, post-hoc): does
         every claim in the generated answer trace back to context?
                    │
              Cited answer, or refusal — either way, explained
                    │
        FastAPI (/ask), Dockerized, eval suite run in CI on every push
```

## Why two separate LLM-judge checks (self-check vs. faithfulness)

These check different things and a model can pass one while failing the other:

- **Self-check** (before generation): "does the retrieved context contain enough to
  answer this question?" Catches wrong-scheme-but-topically-similar retrieval — e.g. a
  PM-JAY question that pulls back PMAY/pension content, which reads confidently
  on-topic (cosine similarity ~0.83) without actually answering the question asked.
- **Faithfulness** (after generation): "did the model's actual answer stick to what
  the context says?" Catches the case where context genuinely supported an answer, but
  generation added an unsupported extra detail anyway.

## The guardrail's design, and why it isn't just a similarity threshold

Real measurements from testing this exact corpus:

| Query type | Similarity |
|---|---|
| Clearly answerable | 0.86 – 0.89 |
| On-topic, but no chunk directly answers it | 0.85 – 0.86 |
| Wrong scheme, same domain (PM-JAY, never ingested) | 0.828 |
| Fully unrelated ("capital of France") | 0.74 – 0.75 |

The gap between "answerable" (0.86) and "wrong scheme" (0.828) is only ~0.03 — too
narrow to safely threshold on. A pure similarity cutoff would either refuse real
questions on unlucky phrasing or confidently answer from the wrong scheme's rules.
The guardrail therefore uses a cheap similarity pre-filter (0.78) only to catch
obvious noise cheaply, and relies on the LLM self-check as the real gate for anything
that clears it.

## Security posture

This system is a solo-built, single-user, local-dev RAG pipeline over manually
collected official PDFs — not a multi-tenant production system serving untrusted
users or third-party document uploads. Reviewed against OWASP's RAG security
guidance and scoped accordingly:

**Implemented:**
- Fail-closed design throughout — low-confidence OCR pages are excluded rather than
  silently indexed; low-similarity or failed-self-check queries are refused rather
  than answered from the model's own knowledge; any pipeline exception returns a
  clear error rather than falling back to an ungrounded answer.
- Retrieved content is wrapped in explicit "this is DATA, not instructions"
  delimiters before being placed in any LLM prompt (context window / injection
  attack mitigation).
- Pinned dependency versions (`requirements.txt`) as a basic supply-chain control.
- Per-claim source citations (scheme, document, page) on every answer.

**Explicitly out of scope, and why:** per-chunk access control, multi-tenant
isolation, index integrity monitoring, cryptographically signed attribution, response
caching with permission scoping. These defend against threats that require a
multi-user, shared-corpus, or third-party-upload system — none of which this project
is. Building them here would be solving a problem this system doesn't have.

## Goals for version2:

- **Latency is not measured or optimized.** The `/ask` endpoint is synchronous, no
  caching, no batching. A reasonable v2 addition, not a v1 claim.
- **No LLM gateway.** Calls Groq directly via its SDK, not through a
  provider-abstraction layer (e.g. LiteLLM). Fine for a single-provider v1; a gateway
  would add multi-provider fallback and centralized rate-limit pooling if this grew.
- **CI does not run PDF ingestion.** GitHub Actions tests against a committed
  `data/processed/chunks.jsonl`, not by re-running OCR on real PDFs — Tesseract setup
  in CI plus large binary PDFs in a repo were judged not worth it for v1. This means
  a bug specifically in the ingestion/OCR layer won't be caught by CI, only by local
  testing before a commit.
- **Free-tier rate limits.** Groq's free tier (8,000 tokens/minute) is easy to hit
  running the full eval set back-to-back; `llm_client.py` retries with exponential
  backoff, and `run_eval.py` paces requests, but this is a real constraint on how the
  eval suite can be run, not an unlimited resource.
- **Chunking can still split a condition from its exception** in rare cases despite
  20% overlap — a known, mitigated-not-solved risk with eligibility text specifically
  (see `app/ingestion/chunker.py` for the detailed reasoning).

## Tech stack

PDF parsing: pdfplumber + Tesseract OCR (eng+hin) · Chunking/orchestration: LangChain
· Embeddings: sentence-transformers (multilingual-e5-base) · Vector store: PostgreSQL
+ pgvector (HNSW, cosine) · LLM: Groq API (openai/gpt-oss-120b) · API: FastAPI ·
Container: Docker / docker-compose · CI: GitHub Actions

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Install Tesseract (with Hindi language pack) and Poppler — see inline comments in
`requirements.txt` for OS-specific instructions.

Copy `.env.example` to `.env` and add your Groq API key (free tier:
console.groq.com/keys).

```bash
docker-compose up -d                    # starts Postgres + pgvector
python scripts/ingest.py                # PDF -> chunks (needs your own PDFs in data/raw/)
python scripts/embed_and_store.py       # chunks -> embeddings -> pgvector
python scripts/query.py "your question" # retrieval only, no generation
python scripts/ask.py "your question"   # full pipeline: retrieve, guardrail, answer
python scripts/run_eval.py              # run the eval suite, prints + writes eval/results.json
```

Or run the API directly:

```bash
uvicorn app.api.main:app --reload
# POST /ask {"question": "..."}
```

## Project structure

```
app/
  ingestion/     PDF loading (with OCR fallback), chunking
  retrieval/     embeddings, pgvector storage, similarity search
  generation/    LLM client, prompts, guardrail, faithfulness checker
  api/           FastAPI app
scripts/         CLI entry points for each pipeline stage
eval/            held-out eval set + results
scripts/init.sql database schema
```
