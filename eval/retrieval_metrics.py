"""
Reference-based retrieval metrics computed over a ranked list of retrieved
chunks whose relevance labels are derived from the reference answer
(see eval/relevance.py).

Relevance is judged within the retrieved candidate pool (the top-N actually
fetched), so recall@k is measured against the relevant items found in that pool
— a standard practical definition when the full set of relevant chunks in the
corpus is unknown.
"""

import math
from typing import List


def precision_at_k(rels: List[bool], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    if k <= 0:
        return 0.0
    topk = rels[:k]
    return sum(1 for r in topk if r) / k


def recall_at_k(rels: List[bool], k: int) -> float:
    """Relevant in top-k over all relevant in the candidate pool.

    Returns NaN-safe 0.0 when there are no relevant items in the pool (caller
    should track how many such cases occur).
    """
    total_relevant = sum(1 for r in rels if r)
    if total_relevant == 0:
        return 0.0
    return sum(1 for r in rels[:k] if r) / total_relevant


def hit_rate(rels: List[bool], k: int) -> float:
    """1.0 if at least one relevant chunk appears in the top-k, else 0.0."""
    return 1.0 if any(rels[:k]) else 0.0


def mrr(rels: List[bool]) -> float:
    """Reciprocal rank of the first relevant chunk."""
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(grades: List[float], k: int) -> float:
    """Normalized DCG using graded relevance in [0,1]."""
    def dcg(gs: List[float]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gs[:k]))

    actual = dcg(grades)
    ideal = dcg(sorted(grades, reverse=True))
    return (actual / ideal) if ideal > 0 else 0.0


def context_recall(answer_sentences: List[str], chunk_texts: List[str], embedder, threshold: float = 0.55) -> float:
    """RAGAS-style: fraction of reference-answer sentences supported by the retrieved context.

    A sentence is "supported" if its max embedding cosine against any retrieved
    chunk clears ``threshold``.
    """
    from eval.relevance import cosine

    if not answer_sentences or not chunk_texts:
        return 0.0
    chunk_embs = [embedder.embed(c) for c in chunk_texts]
    supported = 0
    for sent in answer_sentences:
        s_emb = embedder.embed(sent)
        if any(cosine(s_emb, c_emb) >= threshold for c_emb in chunk_embs):
            supported += 1
    return supported / len(answer_sentences)
