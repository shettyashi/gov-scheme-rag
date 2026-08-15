"""
Phase 5 entry point — the eval harness.

Usage:
    python scripts/run_eval.py

Runs every question in eval/eval_set.jsonl through the full pipeline
(retrieve -> guardrail -> generate/refuse), scores each result, and
prints a report. Writes eval/results.json for Phase 6's CI step to
read and fail the build on regression.

Metrics computed:
- Retrieval hit-rate@k: for answerable questions with a known
  expected_document, did that document appear anywhere in the top-k
  retrieved chunks? (Doesn't require the guardrail to have answered —
  this isolates retrieval quality specifically.)
- Refusal correctness: did the pipeline's answered/refused decision
  match the expected label? Reported as accuracy, but ALSO broken out
  by direction (false answers vs false refusals) because these are
  not equally bad — a false answer is a hallucination risk, a false
  refusal is just unhelpfulness. Collapsing them into one accuracy
  number hides which failure mode you actually have.
- Faithfulness rate: for questions the pipeline actually answered,
  what fraction had every claim traced back to retrieved content
  (via the LLM-judge in app/generation/faithfulness.py)?
- Hallucination rate: 1 - faithfulness rate, reported directly since
  that's the PRD's named target metric (<5%).

Questions with answerable=null (unverified ground truth) are skipped
from scoring and reported separately — see eval_set.jsonl's notes
for which ones and why.
"""

import json
import sys
import time
from pathlib import Path

from app.generation.faithfulness import check_faithfulness
from app.generation.guardrail import answer_question
from app.retrieval.embedder import embed_query
from app.retrieval.store import get_connection, similarity_search

EVAL_SET_PATH = Path("eval/eval_set.jsonl")
RESULTS_PATH = Path("eval/results.json")
TOP_K = 5
PACING_DELAY_SECONDS = 2  # between questions, to stay under Groq free-tier TPM limits

# Thresholds from the v1 PRD - Phase 6's CI step reads results.json and
# fails the build if these aren't met. Not enforced here directly; this
# script just reports the numbers honestly.
TARGET_HIT_RATE = 0.90
TARGET_HALLUCINATION_RATE = 0.05


def load_eval_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(item: dict, conn) -> dict:
    question = item["question"]
    query_vec = embed_query(question)
    chunks = similarity_search(conn, query_vec, k=TOP_K)

    result = answer_question(question, chunks)

    retrieved_docs = [c.document_name for c in chunks]
    hit = item.get("expected_document") in retrieved_docs if item.get("expected_document") else None

    row = {
        "id": item["id"],
        "question": question,
        "expected_answerable": item["answerable"],
        "expected_document": item.get("expected_document"),
        "actual_answered": result.answered,
        "retrieval_hit": hit,
        "top_similarity": result.top_similarity,
        "answer": result.answer,
        "refusal_reason": result.refusal_reason,
    }

    if result.answered:
        faith = check_faithfulness(result.answer, chunks)
        row["faithfulness_status"] = faith.status  # FAITHFUL / UNFAITHFUL / INVALID_JUDGE_RESPONSE
        row["faithful"] = faith.faithful  # True / False / None
        row["faithfulness_reason"] = faith.reason
    else:
        row["faithfulness_status"] = None
        row["faithful"] = None
        row["faithfulness_reason"] = None

    return row


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["expected_answerable"] is not None]
    skipped = [r for r in rows if r["expected_answerable"] is None]

    hit_rows = [r for r in scored if r["expected_document"] is not None]
    hits = [r for r in hit_rows if r["retrieval_hit"]]
    hit_rate = len(hits) / len(hit_rows) if hit_rows else None

    correct_decisions = [r for r in scored if r["actual_answered"] == r["expected_answerable"]]
    refusal_accuracy = len(correct_decisions) / len(scored) if scored else None

    false_answers = [r for r in scored if r["actual_answered"] and not r["expected_answerable"]]
    false_refusals = [r for r in scored if not r["actual_answered"] and r["expected_answerable"]]

    answered_rows = [r for r in scored if r["actual_answered"]]
    invalid_judge_rows = [r for r in answered_rows if r["faithfulness_status"] == "INVALID_JUDGE_RESPONSE"]
    judged_rows = [r for r in answered_rows if r["faithfulness_status"] in ("FAITHFUL", "UNFAITHFUL")]
    faithful_rows = [r for r in judged_rows if r["faithful"] is True]
    # Denominator excludes invalid judge responses on purpose - a judge
    # that failed to explain itself is neither evidence of faithfulness
    # nor of hallucination. Counting it either way would misrepresent
    # what was actually measured. See faithfulness.py for the full
    # reasoning.
    faithfulness_rate = len(faithful_rows) / len(judged_rows) if judged_rows else None
    hallucination_rate = (1 - faithfulness_rate) if faithfulness_rate is not None else None

    return {
        "total_questions": len(rows),
        "scored_questions": len(scored),
        "skipped_unverified": len(skipped),
        "retrieval_hit_rate": hit_rate,
        "retrieval_hit_rate_n": len(hit_rows),
        "refusal_accuracy": refusal_accuracy,
        "false_answers_count": len(false_answers),
        "false_answers_ids": [r["id"] for r in false_answers],
        "false_refusals_count": len(false_refusals),
        "false_refusals_ids": [r["id"] for r in false_refusals],
        "answered_count": len(answered_rows),
        "faithfulness_rate": faithfulness_rate,
        "hallucination_rate": hallucination_rate,
        "invalid_judge_response_count": len(invalid_judge_rows),
        "invalid_judge_response_ids": [r["id"] for r in invalid_judge_rows],
        "meets_hit_rate_target": (hit_rate >= TARGET_HIT_RATE) if hit_rate is not None else None,
        "meets_hallucination_target": (
            hallucination_rate <= TARGET_HALLUCINATION_RATE
        ) if hallucination_rate is not None else None,
    }


def main():
    if not EVAL_SET_PATH.exists():
        print(f"No eval set at {EVAL_SET_PATH}")
        return

    items = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(items)} eval questions.\n")

    conn = get_connection()
    rows = []
    for i, item in enumerate(items):
        print(f"  Running {item['id']}: {item['question'][:60]}...")
        rows.append(run_one(item, conn))
        if i < len(items) - 1:
            time.sleep(PACING_DELAY_SECONDS)
    conn.close()

    summary = summarize(rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print(f"\n{'='*60}")
    print("EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions:        {summary['total_questions']}")
    print(f"Scored (verified):      {summary['scored_questions']}")
    print(f"Skipped (unverified):   {summary['skipped_unverified']}")
    print()
    hr = summary['retrieval_hit_rate']
    print(f"Retrieval hit-rate@{TOP_K}:   "
          f"{hr:.1%} ({summary['retrieval_hit_rate_n']} questions) "
          f"{'[MEETS' if summary['meets_hit_rate_target'] else '[BELOW'} "
          f"{TARGET_HIT_RATE:.0%} target]" if hr is not None else "N/A")
    print(f"Refusal accuracy:       {summary['refusal_accuracy']:.1%}"
          if summary['refusal_accuracy'] is not None else "N/A")
    print(f"  False answers:        {summary['false_answers_count']} {summary['false_answers_ids']}")
    print(f"  False refusals:       {summary['false_refusals_count']} {summary['false_refusals_ids']}")
    hal = summary['hallucination_rate']
    print(f"Hallucination rate:     "
          f"{hal:.1%} {'[MEETS' if summary['meets_hallucination_target'] else '[EXCEEDS'} "
          f"{TARGET_HALLUCINATION_RATE:.0%} target]" if hal is not None else "N/A")
    if summary["invalid_judge_response_count"] > 0:
        print(f"  ⚠ INVALID JUDGE RESPONSES: {summary['invalid_judge_response_count']} "
              f"{summary['invalid_judge_response_ids']} — excluded from hallucination_rate "
              f"above, but the judge failed to give a valid verdict on these. Review manually "
              f"before trusting this run's number.")
    print(f"\nFull results written to {RESULTS_PATH}")

    # Exit nonzero on target miss so CI (Phase 6) actually fails the
    # build rather than just printing a report nobody reads. Targets
    # that were never measurable (e.g. no answered questions at all)
    # don't count as a failure here - that's a different, worse
    # problem (something broke upstream) that would show up as an
    # exception before reaching this point, not a silent pass.
    hit_ok = summary["meets_hit_rate_target"] is not False
    hallucination_ok = summary["meets_hallucination_target"] is not False
    if not (hit_ok and hallucination_ok):
        print("\nFAILED: one or more targets not met.")
        sys.exit(1)


if __name__ == "__main__":
    main()
