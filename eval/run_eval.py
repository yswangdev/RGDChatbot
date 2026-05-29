"""
Orchestrator for the RAG evaluation harness.

Builds a fixed set of answerable QA pairs, then for each question runs
reference-based retrieval + generation metrics, LLM-as-judge scoring, and a
robustness stress test on the SAME questions, combines them into an overall
score, and writes per-question results + an aggregate summary and report.

Example:
  python -m eval.run_eval --n 50 --vector-backend chroma
  python -m eval.run_eval --n 50 --pg-dsn "$PG_DSN" --no-bertscore
"""

import argparse
import json
import os
from statistics import mean

from RAG import RAGSystem
from eval.dataset import load_or_build_eval_set
from eval.relevance import EmbeddingCache, label_relevance, split_sentences
from eval import retrieval_metrics as rm
from eval.generation_metrics import compute_generation_metrics
from eval.llm_judge import LLMJudge
from eval.stress import robustness_score
from eval.scoring import family_scores, overall_score, DEFAULT_WEIGHTS
from eval.report import render_report


def evaluate_item(item, rag, embedder, judge, args):
    """Run all metric families for one QA pair; return a result record."""
    question = item["question"]
    reference = item["reference_answer"]

    # --- Retrieval (pool larger than k for a recall denominator) ---
    pool_n = max(args.k * 2, 10)
    pool = rag.retrieve(question, n_results=pool_n, min_similarity=0.0)
    chunk_texts = [d["document"] for d in pool]
    answer_emb = embedder.embed(reference)

    labels = [label_relevance(reference, c, embedder, answer_emb) for c in chunk_texts]
    rels = [l["relevant"] for l in labels]
    grades = [l["graded"] for l in labels]
    topk_texts = chunk_texts[: args.k]

    retrieval = {
        "precision_at_k": rm.precision_at_k(rels, args.k),
        "recall_at_k": rm.recall_at_k(rels, args.k),
        "hit_rate": rm.hit_rate(rels, args.k),
        "mrr": rm.mrr(rels),
        "ndcg_at_k": rm.ndcg_at_k(grades, args.k),
        "context_recall": rm.context_recall(split_sentences(reference), topk_texts, embedder),
    }

    # --- Generation ---
    candidate = rag.generate(
        question, n_results=args.k, min_similarity=args.min_similarity, include_sources=False,
        temperature=0.0, seed=0,
    )
    generation = compute_generation_metrics(candidate, reference, use_bertscore=not args.no_bertscore)

    # --- LLM judge ---
    judge_retrieval = {}
    judge_generation = {}
    if not args.no_judge:
        chunk_scores = [judge.judge_chunk_relevance(question, c) for c in topk_texts]
        judge_retrieval = {
            "chunk_relevance_avg": mean(chunk_scores) if chunk_scores else 0,
            "context_quality": judge.judge_context_quality(question, topk_texts),
        }
        context_joined = "\n\n".join(topk_texts)
        judge_generation = judge.judge_generation(question, context_joined, candidate, reference)

    # --- Robustness ---
    robustness = {"stability": 0.0, "variants": {}}
    if not args.no_stress:
        gen_fn = lambda q: rag.generate(
            q, n_results=args.k, min_similarity=args.min_similarity, include_sources=False,
            temperature=0.0, seed=0,
        )
        robustness = robustness_score(question, candidate, gen_fn, embedder, seed=args.seed)

    record = {
        "id": item["id"],
        "type": item.get("type", ""),
        "question": question,
        "reference_answer": reference,
        "rag_answer": candidate,
        "retrieval": retrieval,
        "generation": generation,
        "judge_retrieval": judge_retrieval,
        "judge_generation": judge_generation,
        "robustness": robustness,
    }
    record["families"] = family_scores(record)
    record["overall"] = overall_score(record["families"], args.weights)
    return record


def aggregate(records, weights):
    """Compute system-level means across all records."""
    if not records:
        return {}
    families = list(DEFAULT_WEIGHTS.keys())
    fam_means = {f: mean(r["families"][f] for r in records) for f in families}
    summary = {
        "n": len(records),
        "weights": weights,
        "family_means": fam_means,
        "overall": mean(r["overall"] for r in records),
        "by_type": {},
    }
    types = sorted({r["type"] for r in records})
    for t in types:
        subset = [r for r in records if r["type"] == t]
        summary["by_type"][t] = {
            "n": len(subset),
            "overall": mean(r["overall"] for r in subset),
        }
    return summary


def parse_weights(s):
    if not s:
        return DEFAULT_WEIGHTS
    weights = dict(DEFAULT_WEIGHTS)
    for pair in s.split(","):
        k, v = pair.split("=")
        weights[k.strip()] = float(v)
    return weights


def main():
    p = argparse.ArgumentParser(description="RAG evaluation harness")
    p.add_argument("--qa-report", default="qa_report.html")
    p.add_argument("--out", default="data/eval")
    p.add_argument("--n", type=int, default=50, help="Eval set size")
    p.add_argument("--limit", type=int, default=None, help="Cap items evaluated (debug)")
    p.add_argument("--k", type=int, default=5, help="Top-k for retrieval/generation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild-set", action="store_true", help="Rebuild the cached eval set")
    # RAG config (mirrors rag_cli.py)
    p.add_argument("--vector-backend", choices=["pgvector", "chroma"], default="pgvector")
    p.add_argument("--pg-dsn", default=os.environ.get("PG_DSN"))
    p.add_argument("--persist-dir", default="./chroma_db")
    p.add_argument("--collection", default="rgd_chunks")
    p.add_argument("--embedding-host", default="http://grudge.rgd.mcw.edu:11434")
    p.add_argument("--llm-host", default="http://grudge.rgd.mcw.edu:11434")
    p.add_argument("--embedding-model", default="mxbai-embed-large:latest")
    p.add_argument("--llm-model", default="llama3.2")
    p.add_argument("--min-similarity", type=float, default=0.3)
    # Judge / metric toggles
    p.add_argument("--judge-model", default="qwen2.5:7b")
    p.add_argument("--judge-host", default="http://grudge.rgd.mcw.edu:11434")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-stress", action="store_true")
    p.add_argument("--weights", default=None, help="e.g. judge_generation=0.4,robustness=0.05")
    args = p.parse_args()
    args.weights = parse_weights(args.weights)

    rag = RAGSystem(
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        embedding_host=args.embedding_host,
        llm_host=args.llm_host,
        llm_model=args.llm_model,
        persist_directory=args.persist_dir,
        vector_backend=args.vector_backend,
        pg_dsn=args.pg_dsn,
    )
    if rag.store.count() == 0:
        print("Vector store is empty — run `python rag_cli.py --build` first.")
        return

    os.makedirs(args.out, exist_ok=True)
    eval_set = load_or_build_eval_set(
        args.qa_report, rag,
        out_path=os.path.join(args.out, "eval_set.jsonl"),
        n=args.n, seed=args.seed, rebuild=args.rebuild_set,
    )
    if args.limit:
        eval_set = eval_set[: args.limit]
    print(f"Evaluating {len(eval_set)} question(s) at k={args.k}...")

    embedder = EmbeddingCache(rag._get_embedding, os.path.join(args.out, "emb_cache.json"))
    judge = LLMJudge(args.judge_host, args.judge_model)

    records = []
    results_path = os.path.join(args.out, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(eval_set, 1):
            print(f"  [{i}/{len(eval_set)}] #{item['id']} {item['question'][:60]}...")
            rec = evaluate_item(item, rag, embedder, judge, args)
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            embedder.save()

    summary = aggregate(records, args.weights)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    render_report(records, summary, args.out)
    print(f"\nSystem overall: {summary['overall']:.3f} over {summary['n']} questions")
    print(f"Report: {os.path.join(args.out, 'eval_report.html')}")
    print(f"Table:  {os.path.join(args.out, 'eval_results.csv')}")


if __name__ == "__main__":
    main()
