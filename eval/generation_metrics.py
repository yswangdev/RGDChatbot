"""
Reference-based generation metrics: compare the RAG answer (candidate) to the
cleaned human reference answer.

The lexical metrics are deliberately trimmed to a non-redundant pair — BLEU /
ROUGE-1 / ROUGE-2 / token-F1 all measure the same n-gram overlap and correlate
heavily, so we keep only:

- ROUGE-L: longest-common-subsequence overlap (word order / structure).
- METEOR: stem- and synonym-aware overlap (the only lexically flexible one).
- BERTScore: semantic similarity (paraphrase-tolerant); lazily loaded, skip with
  use_bertscore=False to avoid the torch/transformers download.
"""

from typing import Dict

from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_NLTK_READY = False
_BERT_SCORER = None


def _ensure_nltk() -> bool:
    """Lazily ensure NLTK METEOR resources; return False if unavailable."""
    global _NLTK_READY
    if _NLTK_READY:
        return True
    try:
        import nltk

        for res in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)
        _NLTK_READY = True
    except Exception:
        _NLTK_READY = False
    return _NLTK_READY


def _meteor(candidate: str, reference: str) -> float:
    if not _ensure_nltk():
        return 0.0
    try:
        from nltk.translate.meteor_score import meteor_score

        return float(meteor_score([reference.split()], candidate.split()))
    except Exception:
        return 0.0


def _bertscore(candidate: str, reference: str) -> float:
    global _BERT_SCORER
    try:
        from bert_score import BERTScorer

        if _BERT_SCORER is None:
            _BERT_SCORER = BERTScorer(lang="en", rescale_with_baseline=True)
        _, _, f1 = _BERT_SCORER.score([candidate], [reference])
        return float(f1.mean())
    except Exception as e:
        print(f"  (bertscore unavailable: {e})")
        return 0.0


def compute_generation_metrics(candidate: str, reference: str, use_bertscore: bool = True) -> Dict[str, float]:
    """Return reference-based generation scores, all normalized to [0,1].

    Trimmed to non-redundant aspects: ROUGE-L (structure), METEOR (flexible
    lexical), and BERTScore (semantic).
    """
    candidate = candidate or ""
    reference = reference or ""

    metrics = {
        "rougeL": _ROUGE.score(reference, candidate)["rougeL"].fmeasure,
        "meteor": _meteor(candidate, reference),
    }
    if use_bertscore:
        metrics["bertscore"] = _bertscore(candidate, reference)
    return metrics
