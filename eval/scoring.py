"""
Score normalization and combination.

Every sub-metric is mapped to [0,1], grouped into five families, and combined
into a single overall score via a configurable weighted average — per question
and (by averaging) for the whole system.
"""

from typing import Dict

# Default family weights (sum to 1.0). Override via run_eval --weights.
DEFAULT_WEIGHTS = {
    "retrieval_ref": 0.20,
    "generation_ref": 0.20,
    "judge_retrieval": 0.15,
    "judge_generation": 0.35,
    "robustness": 0.10,
}


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def norm_judge_5(x: float) -> float:
    """Map a 1-5 judge score to [0,1]."""
    return (float(x) - 1.0) / 4.0


def norm_judge_3(x: float) -> float:
    """Map a 0-3 relevance score to [0,1]."""
    return float(x) / 3.0


def family_scores(record: Dict) -> Dict[str, float]:
    """Collapse a per-question result record into the five family scores in [0,1].

    Expects keys: retrieval (metric dict), generation (metric dict),
    judge_retrieval {chunk_relevance_avg (0-3), context_quality (1-5)},
    judge_generation {faithfulness, correctness, answer_relevance, completeness (1-5)},
    robustness {stability (0-1)}.
    """
    # hit_rate dropped: it is mathematically redundant (hit_rate=1 iff MRR>0).
    r = record.get("retrieval", {})
    retrieval_ref = _mean([
        r.get("precision_at_k"), r.get("recall_at_k"),
        r.get("mrr"), r.get("ndcg_at_k"), r.get("context_recall"),
    ])

    # Trimmed to non-redundant aspects: ROUGE-L, METEOR, BERTScore.
    # BERTScore is baseline-rescaled (can be negative; ~0 == baseline), so clamp
    # to [0,1] before averaging into the normalized family score.
    g = record.get("generation", {})
    bert = g.get("bertscore")
    bert = max(0.0, bert) if isinstance(bert, (int, float)) else None
    generation_ref = _mean([g.get("rougeL"), g.get("meteor"), bert])

    jr = record.get("judge_retrieval", {})
    judge_retrieval = _mean([
        norm_judge_3(jr.get("chunk_relevance_avg", 0)),
        norm_judge_5(jr.get("context_quality", 1)),
    ])

    jg = record.get("judge_generation", {})
    judge_generation = _mean([
        norm_judge_5(jg.get("faithfulness", 1)),
        norm_judge_5(jg.get("correctness", 1)),
        norm_judge_5(jg.get("answer_relevance", 1)),
        norm_judge_5(jg.get("completeness", 1)),
    ])

    robustness = float(record.get("robustness", {}).get("stability", 0.0))

    return {
        "retrieval_ref": retrieval_ref,
        "generation_ref": generation_ref,
        "judge_retrieval": judge_retrieval,
        "judge_generation": judge_generation,
        "robustness": robustness,
    }


def overall_score(families: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Weighted average of the family scores."""
    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(weights.get(k, 0) for k in families)
    if total_w == 0:
        return 0.0
    return sum(families[k] * weights.get(k, 0) for k in families) / total_w
