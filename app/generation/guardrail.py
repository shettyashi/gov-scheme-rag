"""
Confidence guardrail — the core differentiator of this project.

Two-stage design, based on real measurements from Phase 3 testing on
this exact corpus (see README / eval notes for the actual numbers):

Stage 1 — cheap similarity pre-filter (SIMILARITY_FLOOR):
Fully unrelated queries ("what is the capital of France?") scored
~0.74-0.75 cosine similarity against this corpus. Answerable queries
scored 0.85+. A conservative floor around 0.78-0.80 catches obvious
nonsense without spending an LLM call on it — cheap, fast, saves
quota on Groq's free tier.

Stage 2 — LLM self-check (the one that actually matters):
Measured gap between a genuinely answerable query (0.85-0.89) and a
wrong-scheme-but-same-domain query (PM-JAY, never ingested, scored
0.828) was only ~0.03 — too narrow to trust a similarity threshold
alone. The self-check asks the model directly: does this retrieved
content actually answer the question, not just relate to it? This is
what catches "confidently retrieved the wrong scheme's eligibility
rules" — the failure mode similarity alone can't distinguish.

Both stages must pass for an answer to be generated. Either failing
triggers a refusal with a stated reason — fail-closed, not fail-open.
"""

from dataclasses import dataclass

from app.generation.llm_client import chat
from app.generation.prompt import (
    GENERATION_SYSTEM_PROMPT,
    SELF_CHECK_SYSTEM_PROMPT,
    format_chunks_for_prompt,
)
from app.retrieval.store import RetrievedChunk

SIMILARITY_FLOOR = 0.78  # tune against your own eval set once built in Phase 5


@dataclass
class GuardrailResult:
    answered: bool
    answer: str | None
    refusal_reason: str | None
    top_similarity: float
    self_check_passed: bool | None  # None if never reached (failed pre-filter first)


def _self_check(question: str, chunks: list[RetrievedChunk]) -> tuple[bool, str]:
    context_block = format_chunks_for_prompt(chunks)
    user_prompt = f"Question: {question}\n\n{context_block}"
    response = chat(SELF_CHECK_SYSTEM_PROMPT, user_prompt, temperature=0.0)

    lines = response.strip().split("\n", 1)
    verdict = lines[0].strip().upper()
    reason = lines[1].strip() if len(lines) > 1 else ""
    return verdict.startswith("YES"), reason


def _generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    context_block = format_chunks_for_prompt(chunks)
    user_prompt = f"Question: {question}\n\n{context_block}"
    return chat(GENERATION_SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=1024)


def answer_question(question: str, chunks: list[RetrievedChunk]) -> GuardrailResult:
    if not chunks:
        return GuardrailResult(
            answered=False, answer=None,
            refusal_reason="No documents retrieved for this question.",
            top_similarity=0.0, self_check_passed=None,
        )

    top_similarity = chunks[0].similarity

    if top_similarity < SIMILARITY_FLOOR:
        return GuardrailResult(
            answered=False, answer=None,
            refusal_reason=(
                f"Retrieved content isn't similar enough to the question "
                f"(top similarity {top_similarity:.3f}, floor {SIMILARITY_FLOOR}) "
                f"— this question may be about a scheme or topic not covered "
                f"by the available documents."
            ),
            top_similarity=top_similarity, self_check_passed=None,
        )

    can_answer, reason = _self_check(question, chunks)
    if not can_answer:
        return GuardrailResult(
            answered=False, answer=None,
            refusal_reason=(
                f"Retrieved content is topically related but doesn't "
                f"directly answer this question. {reason}"
            ),
            top_similarity=top_similarity, self_check_passed=False,
        )

    answer = _generate_answer(question, chunks)
    return GuardrailResult(
        answered=True, answer=answer, refusal_reason=None,
        top_similarity=top_similarity, self_check_passed=True,
    )
