"""
Recompute generation metrics (incl. BERTScore) and overall scores from an
existing results.jsonl WITHOUT re-running retrieval / the LLM judge.

Useful after changing the metric set or enabling BERTScore: it reuses the
already-generated answers (rag_answer / reference_answer) in each record.

  python -m eval.recompute --out data/eval
"""

import argparse
import json
import os
from statistics import mean

from eval.generation_metrics import compute_generation_metrics
from eval.scoring import family_scores, overall_score, DEFAULT_WEIGHTS


def aggregate(records, weights):
    families = list(DEFAULT_WEIGHTS.keys())
    fam_means = {f: mean(r["families"][f] for r in records) for f in families}
    summary = {
        "n": len(records),
        "weights": weights,
        "family_means": fam_means,
        "overall": mean(r["overall"] for r in records),
        "by_type": {},
    }
    for t in sorted({r["type"] for r in records}):
        subset = [r for r in records if r["type"] == t]
        summary["by_type"][t] = {"n": len(subset), "overall": mean(r["overall"] for r in subset)}
    return summary


def main():
    p = argparse.ArgumentParser(description="Recompute generation metrics + scores from results.jsonl")
    p.add_argument("--out", default="data/eval")
    p.add_argument("--no-bertscore", action="store_true")
    args = p.parse_args()

    results_path = os.path.join(args.out, "results.jsonl")
    records = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]
    print(f"Recomputing generation metrics for {len(records)} records "
          f"(bertscore={'off' if args.no_bertscore else 'on'})...")

    for i, r in enumerate(records, 1):
        r["generation"] = compute_generation_metrics(
            r.get("rag_answer", ""), r.get("reference_answer", ""),
            use_bertscore=not args.no_bertscore,
        )
        r.get("retrieval", {}).pop("hit_rate", None)  # drop redundant metric
        r["families"] = family_scores(r)
        r["overall"] = overall_score(r["families"], DEFAULT_WEIGHTS)
        if i % 10 == 0:
            print(f"  {i}/{len(records)}")

    with open(results_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate(records, DEFAULT_WEIGHTS)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    from eval.report import render_report
    render_report(records, summary, args.out)
    print(f"System overall: {summary['overall']:.3f} over {summary['n']} questions")


if __name__ == "__main__":
    main()
