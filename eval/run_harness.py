"""Eval harness for the Corrective Hybrid RAG Assistant.

v3: fixes a context-truncation bug from v2. chain.py's retrieve() builds
TWO separate representations of each retrieved chunk:
  - context_parts: FULL chunk text with a "[Document: X, page Y]" prefix —
    this is what actually gets sent to the LLM for generation AND is what
    verify_answer() should be checking against.
  - sources[i]["content"]: a 200-char TRUNCATED preview, meant only for
    UI display in app.py — NOT what the model actually saw.

v2 mistakenly built its "contexts" field from `sources`, so the verifier
(and RAGAS) were judging answers against a truncated, prefix-less snippet
instead of the real context — producing an artificially high FAIL rate.
v3 calls retrieve() once, keeps context_parts, and reuses chain.py's own
_build_messages()/RAG_SYSTEM_PROMPT so generation exactly matches what
ask() does in production, without a second (costly) retrieval pass.

Also retains v2's rate-limit handling:
  - retry with exponential backoff on 429 errors
  - throttling delay between questions
  - incremental checkpointing (--resume to continue a partial run)

Usage:
    python eval/run_harness.py --dataset eval/eval_dataset.json --out eval/harness_results.json
    python eval/run_harness.py --resume            # skip questions already in --out
    python eval/run_harness.py --delay 20          # seconds between questions

IMPORTANT: if you have a harness_results.json from v2 of this script, it
used truncated context — do NOT --resume from it. Run fresh (a different
--out path, or delete the old file first) so every question is redone
with the corrected full-context logic.

Prerequisites:
    - You've already uploaded your PDFs through the Streamlit app at least
      once, so a `faiss_index/` folder exists on disk (see config.FAISS_INDEX_PATH).
    - Your .env has GROQ_API_KEY set (same as running the app).
"""

import os
import re
import sys
import json
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import load_index
from rag.bm25 import BM25Index
from rag.chain import retrieve, _build_messages, RAG_SYSTEM_PROMPT, _fallback
from rag.llm import get_llm
from rag.answer_verifier import verify_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
DEFAULT_BACKOFF_SEC = 30

_WAIT_HINT_RE = re.compile(r"try again in (\d+)m([\d.]+)s|try again in ([\d.]+)s")


def _parse_wait_hint(error_message: str) -> float:
    match = _WAIT_HINT_RE.search(error_message)
    if not match:
        return DEFAULT_BACKOFF_SEC
    if match.group(1) is not None:
        return int(match.group(1)) * 60 + float(match.group(2))
    return float(match.group(3))


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg


def load_pipeline_index():
    vector_store = load_index()
    if vector_store is None:
        raise RuntimeError(
            "No FAISS index found on disk. Upload your PDFs through the "
            "Streamlit app first (streamlit run app.py) so faiss_index/ exists."
        )
    docs = list(vector_store.docstore._dict.values())
    bm25_index = BM25Index(docs)
    logger.info(f"Loaded index with {len(docs)} chunks.")
    return vector_store, bm25_index


def run_one_with_retry(item, vector_store, bm25_index):
    question = item["question"]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _run_one(item, question, vector_store, bm25_index)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                wait = _parse_wait_hint(str(e))
                logger.warning(
                    f"[{item.get('id', '?')}] Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                    f"Waiting {wait:.1f}s before retry..."
                )
                time.sleep(wait)
                continue
            logger.error(f"[{item.get('id', '?')}] FAILED: {e}")
            return {
                "id": item.get("id"),
                "bucket": item.get("bucket"),
                "expected_source": item.get("expected_source"),
                "question": question,
                "ground_truth": item.get("ground_truth"),
                "error": str(e),
            }


def _run_one(item, question, vector_store, bm25_index):
    logger.info(f"[{item.get('id', '?')}] Running: {question[:80]}...")

    # Single retrieval call — reused for both generation and verification,
    # exactly mirroring what ask() does internally (avoids a second,
    # costly grading pass).
    context_parts, sources, confidence, debug_info = retrieve(question, vector_store, bm25_index)

    grades = [
        {
            "source": c.get("source"),
            "label": c.get("grade", {}).get("label") if isinstance(c.get("grade"), dict) else c.get("grade"),
        }
        for c in debug_info.get("chunks", [])
    ]

    if not context_parts:
        # Matches ask()'s fallback branch exactly
        answer = _fallback(question)
        return {
            "id": item.get("id"),
            "bucket": item.get("bucket"),
            "expected_source": item.get("expected_source"),
            "question": question,
            "ground_truth": item.get("ground_truth"),
            "answer": answer,
            "contexts": [],
            "confidence_score": 0.0,
            "confidence_label": "Low",
            "is_fallback": True,
            "num_chunks_retrieved": 0,
            "grades": grades,
            "verification_status": None,
        }

    llm = get_llm()
    messages = _build_messages(RAG_SYSTEM_PROMPT, question, context_parts)
    response = llm.invoke(messages)
    answer = response.content

    # Full context_parts — NOT the truncated sources[i]["content"] — so the
    # verifier sees exactly what the generator saw.
    try:
        verification = verify_answer(question, context_parts, answer)
    except Exception as e:
        if _is_rate_limit_error(e):
            raise
        logger.error(f"Verifier failed for '{question[:50]}...': {e}")
        verification = "ERROR"

    return {
        "id": item.get("id"),
        "bucket": item.get("bucket"),
        "expected_source": item.get("expected_source"),
        "question": question,
        "ground_truth": item.get("ground_truth"),
        "answer": answer,
        "contexts": context_parts,  # full text, matches generation input
        "confidence_score": confidence[0] if confidence else 0.0,
        "confidence_label": confidence[1] if confidence else None,
        "is_fallback": False,
        "num_chunks_retrieved": debug_info.get("num_chunks_retrieved", len(sources)),
        "grades": grades,
        "verification_status": verification,
    }


def load_existing_results(out_path):
    if not os.path.exists(out_path):
        return {}
    with open(out_path, "r") as f:
        existing = json.load(f)
    return {r["id"]: r for r in existing if r.get("id") and "error" not in r}


def save_results(results_by_id, eval_items, out_path):
    ordered = [results_by_id[item["id"]] for item in eval_items if item["id"] in results_by_id]
    with open(out_path, "w") as f:
        json.dump(ordered, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/eval_dataset.json")
    parser.add_argument("--out", default="eval/harness_results.json")
    parser.add_argument("--delay", type=float, default=20.0,
                         help="Seconds to wait between questions (default 20).")
    parser.add_argument("--resume", action="store_true",
                         help="Skip questions that already succeeded in --out from a prior run. "
                              "Do NOT use this with a v2-generated file (truncated context bug) — "
                              "run fresh instead.")
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        eval_items = json.load(f)
    logger.info(f"Loaded {len(eval_items)} eval questions from {args.dataset}")

    vector_store, bm25_index = load_pipeline_index()

    results_by_id = load_existing_results(args.out) if args.resume else {}
    if results_by_id:
        logger.info(f"Resuming: {len(results_by_id)} questions already completed, skipping those.")

    for i, item in enumerate(eval_items):
        if item["id"] in results_by_id:
            continue
        result = run_one_with_retry(item, vector_store, bm25_index)
        results_by_id[item["id"]] = result
        save_results(results_by_id, eval_items, args.out)

        is_last = i == len(eval_items) - 1
        if not is_last and args.delay > 0:
            time.sleep(args.delay)

    logger.info(f"Wrote {len(results_by_id)} results to {args.out}")

    results = [results_by_id[item["id"]] for item in eval_items if item["id"] in results_by_id]
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n⚠️  {len(errors)} question(s) still failed after retries — "
              f"re-run with --resume once fixed:")
        for e in errors:
            print(f"  - {e['id']}: {e['error'][:120]}")

    print("\n--- Harness Run Summary (by bucket) ---")
    buckets = {}
    for r in results:
        if "error" in r:
            continue
        b = r.get("bucket", "unknown")
        buckets.setdefault(b, {"total": 0, "fallback": 0, "verify_fail": 0, "avg_conf": []})
        buckets[b]["total"] += 1
        if r.get("is_fallback"):
            buckets[b]["fallback"] += 1
        if r.get("verification_status") == "FAIL":
            buckets[b]["verify_fail"] += 1
        if "confidence_score" in r:
            buckets[b]["avg_conf"].append(r["confidence_score"])

    for b, stats in buckets.items():
        avg_conf = sum(stats["avg_conf"]) / len(stats["avg_conf"]) if stats["avg_conf"] else 0.0
        print(
            f"{b:15s} | n={stats['total']:2d} | "
            f"fallback={stats['fallback']:2d} | "
            f"verify_fail={stats['verify_fail']:2d} | "
            f"avg_confidence={avg_conf:.1f}%"
        )

    print(f"\nFor 'unanswerable' bucket, fallback count SHOULD be close to n.")
    print(f"For 'single_doc'/'comparison' buckets, fallback should be near 0.")
    print(f"\nNext: python eval/run_ragas.py")


if __name__ == "__main__":
    main()
