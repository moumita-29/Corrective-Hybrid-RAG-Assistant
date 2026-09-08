"""Runs RAGAS scoring on harness_results.json, broken down by bucket.

Usage:
    python eval/run_ragas.py --results eval/harness_results.json

Excludes 'unanswerable' items from RAGAS by default (they have no contexts/
answer to score faithfulness against) but reports fallback accuracy for them
separately, since that's the metric that actually matters for that bucket.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import run_evaluation


def to_ragas_format(item):
    return {
        "question": item["question"],
        "answer": item.get("answer") or "",
        "contexts": item.get("contexts") or [],
        "ground_truth": item.get("ground_truth") or "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval/harness_results.json")
    args = parser.parse_args()

    with open(args.results, "r") as f:
        all_results = json.load(f)

    buckets = {}
    for r in all_results:
        buckets.setdefault(r.get("bucket", "unknown"), []).append(r)

    # --- Unanswerable bucket: report routing accuracy, not RAGAS ---
    if "unanswerable" in buckets:
        items = buckets["unanswerable"]
        correctly_declined = sum(1 for i in items if i.get("is_fallback"))
        print(f"\n=== Bucket: unanswerable ===")
        print(f"Routing accuracy: {correctly_declined}/{len(items)} correctly "
              f"triggered fallback (declined to answer from documents)")
        print("(RAGAS faithfulness/relevancy don't apply here — there's no "
              "grounded answer to score.)")

    # --- Answerable buckets: run RAGAS per bucket ---
    for bucket_name in ("single_doc", "comparison"):
        if bucket_name not in buckets:
            continue
        items = [r for r in buckets[bucket_name] if r.get("contexts")]
        skipped = len(buckets[bucket_name]) - len(items)
        print(f"\n=== Bucket: {bucket_name} ({len(items)} scored"
              + (f", {skipped} skipped — no contexts retrieved)" if skipped else ")"))
        if not items:
            print("No items with retrieved contexts — nothing to score.")
            continue
        ragas_data = [to_ragas_format(i) for i in items]
        run_evaluation(ragas_data)

    print("\n=== Done ===")
    print("Interpretation reminder: a good faithfulness/relevancy score on "
          "single_doc/comparison alongside near-100% routing accuracy on "
          "unanswerable is the story to report — not one blended number.")


if __name__ == "__main__":
    main()
