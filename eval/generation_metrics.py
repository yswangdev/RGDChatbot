"""
Reference-based generation metrics: compare the RAG answer (candidate) to the
cleaned human reference answer.

Lexical: BLEU, ROUGE-1/2/L, METEOR, token-F1 (all in [0,1]).
Semantic: BERTScore F1 (lazily loaded; skip with use_bertscore=False to avoid
the torch/transformers download).
"""

from typing import Dict, Optional

import sacrebleu
from rouge_score import rouge_scorer

from eval.relevance import token_f1

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
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
    """Return reference-based generation scores, all normalized to [0,1]."""
    candidate = candidate or ""
    reference = reference or ""

    bleu = sacrebleu.sentence_bleu(candidate, [reference]).score / 100.0
    rouge = _ROUGE.score(reference, candidate)
    metrics = {
        "bleu": bleu,
        "rouge1": rouge["rouge1"].fmeasure,
        "rouge2": rouge["rouge2"].fmeasure,
        "rougeL": rouge["rougeL"].fmeasure,
        "meteor": _meteor(candidate, reference),
        "token_f1": token_f1(reference, candidate),
    }
    if use_bertscore:
        metrics["bertscore"] = _bertscore(candidate, reference)
    return metrics
