"""
Answer-grounded relevance labeling.

Since the QA report has no gold chunk IDs, a retrieved chunk is judged
"relevant" to a question by how well it matches that question's *reference
answer* — lexically (ROUGE-L overlap) and/or semantically (embedding cosine).
These labels feed the reference-based retrieval metrics.
"""

import hashlib
import json
import os
import re
from typing import Callable, Dict, List, Optional

from rouge_score import rouge_scorer

_TOKEN_RE = re.compile(r"\w+")
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# A chunk counts as relevant if either signal clears its threshold.
LEXICAL_THRESHOLD = 0.18
SEMANTIC_THRESHOLD = 0.55


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def token_f1(a: str, b: str) -> float:
    """Token-overlap F1 between two strings."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    common = sa & sb
    if not common:
        return 0.0
    precision = len(common) / len(sb)
    recall = len(common) / len(sa)
    return 2 * precision * recall / (precision + recall)


def lexical_relevance(answer: str, chunk: str) -> float:
    """Lexical similarity of a chunk to the reference answer (ROUGE-L F)."""
    if not answer or not chunk:
        return 0.0
    return _ROUGE.score(answer, chunk)["rougeL"].fmeasure


def cosine(a: List[float], b: List[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


class EmbeddingCache:
    """Disk-backed cache around an embedder callable (e.g. RAGSystem._get_embedding)."""

    def __init__(self, embed_fn: Callable[[str], List[float]], path: str):
        self.embed_fn = embed_fn
        self.path = path
        self.cache: Dict[str, List[float]] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def embed(self, text: str) -> List[float]:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.embed_fn(text)
        return self.cache[key]

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)


def label_relevance(
    answer: str,
    chunk: str,
    embedder: Optional[EmbeddingCache] = None,
    answer_emb: Optional[List[float]] = None,
) -> Dict:
    """Return relevance signals for one chunk: lexical, semantic, graded, is_relevant."""
    lex = lexical_relevance(answer, chunk)
    sem = 0.0
    if embedder is not None and answer_emb is not None:
        sem = cosine(answer_emb, embedder.embed(chunk))
    graded = max(sem, lex)  # graded relevance in [0,1] for NDCG
    is_rel = lex >= LEXICAL_THRESHOLD or sem >= SEMANTIC_THRESHOLD
    return {"lexical": lex, "semantic": sem, "graded": graded, "relevant": is_rel}


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter for RAGAS-style context recall."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]
