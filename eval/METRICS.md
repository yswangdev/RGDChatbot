# RAG Evaluation — Metrics Reference

This document defines every metric in the harness: what it measures, its range,
how to read each level, and the exact formulas used for normalization and the
overall score. All metrics are computed on the **same** 50 answerable questions.

Two evaluation *methods* are applied to two *stages*:

| | Retrieval | Generation |
|---|---|---|
| **Reference-based** | P@k, R@k, MRR, NDCG@k, context-recall | ROUGE-L, METEOR, BERTScore |
| **LLM-as-judge** (`qwen2.5:7b`) | chunk relevance, context quality | faithfulness, correctness, relevance, completeness |
| **Stress** | — | robustness (answer stability) |

---

## 1. Reference-based retrieval

Relevance has no gold chunk labels, so a retrieved chunk is labeled **relevant
to the question via its reference answer** (see `eval/relevance.py`):

```
lexical  = ROUGE-L F(reference_answer, chunk)          # word-overlap, [0,1]
semantic = cosine(embed(reference_answer), embed(chunk))   # [0,1]
relevant = (lexical >= 0.18) OR (semantic >= 0.55)     # binary label
graded   = max(semantic, lexical)                      # graded relevance [0,1]
```

Metrics are computed over the retrieved candidate pool (top-N, N = max(2k,10)),
with the recall denominator = relevant items found in that pool.

| Metric | Range | Meaning | Formula |
|---|---|---|---|
| **precision@k** | 0–1 | Fraction of the top-k chunks that are relevant. 1.0 = every retrieved chunk is on-topic. | `#relevant(top-k) / k` |
| **recall@k** | 0–1 | Of all relevant chunks in the pool, how many made the top-k. | `#relevant(top-k) / #relevant(pool)` |
| **MRR** | 0–1 | Reciprocal rank of the *first* relevant chunk; rewards putting a good chunk first. 1.0 = first chunk relevant, 0.5 = second, 0 = none. | `1 / rank_first_relevant` |
| **NDCG@k** | 0–1 | Ranking quality using *graded* relevance, discounted by position. 1.0 = best chunks ranked first. | `DCG@k / IDCG@k`, `DCG = Σ grade_i / log2(i+1)` |
| **context-recall** | 0–1 | RAGAS-style: fraction of reference-answer **sentences** semantically supported by the top-k. Measures "did we retrieve enough to cover the answer." | `#supported_sentences / #sentences`, supported if `max cosine(sentence, chunk) ≥ 0.55` |

> **Caveat:** because relevance is judged against the *reference answer* (which
> shares generic RGD vocabulary), these labels are **lenient** — precision/MRR/
> NDCG run high and should be read together with the stricter LLM-judge
> retrieval scores below.

---

## 2. Reference-based generation

Compares the RAG answer (candidate) to the cleaned human reference answer.
Trimmed to three non-redundant aspects (BLEU / ROUGE-1/2 / token-F1 were dropped
as duplicates of these).

| Metric | Range | Aspect | Notes |
|---|---|---|---|
| **ROUGE-L** | 0–1 | Longest-common-subsequence overlap → word order / structure. | Higher = more shared ordered wording. |
| **METEOR** | 0–1 | Unigram match with **stemming + synonyms** + word-order penalty. | The only lexically *flexible* metric; tolerates paraphrase better than ROUGE. |
| **BERTScore** | ~ −1 to 1 | **Semantic** similarity of contextual embeddings (RoBERTa-large), **baseline-rescaled**. | ~0 = baseline (random) similarity; negative = below baseline; >0.3 = clearly similar. Can be negative — see clamping below. |

> **Reading low lexical scores:** human reference answers are long help-desk
> emails. A correct but concise RAG answer will score low on ROUGE/METEOR and
> near-baseline on BERTScore. These are a **sanity floor**, not the verdict —
> the LLM-judge correctness/faithfulness scores are the trustworthy generation
> signal.

---

## 3. LLM-as-judge — retrieval

Judge model `qwen2.5:7b` scores retrieval against the **question** (stricter
than the answer-grounded labels above).

**Chunk relevance — scale 0–3** (averaged over the top-k chunks):

| Score | Meaning |
|---|---|
| 0 | Irrelevant — passage has nothing to do with the question. |
| 1 | Slightly relevant — tangential / shares a topic but doesn't help answer. |
| 2 | Relevant — useful information toward the answer. |
| 3 | Highly relevant — directly answers the question. |

**Context quality — scale 1–5** (the top-k as a whole):

| Score | Meaning |
|---|---|
| 1 | Useless — off-topic, cannot answer the question at all. |
| 2 | Minimal — mostly insufficient, only fragments related. |
| 3 | Partial — some useful info but gaps remain. |
| 4 | Mostly sufficient — answerable with minor gaps. |
| 5 | Fully sufficient — the passages clearly contain the answer. |

---

## 4. LLM-as-judge — generation

Each dimension is scored **1–5** by the judge given the question, retrieved
context, the RAG answer, and the reference answer. A short `rationale` is saved.

**Faithfulness** — is every claim grounded in the retrieved context (no hallucination)?

| 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Fabricated / unsupported | Mostly unsupported | Partly supported, some invented claims | Mostly grounded, minor unsupported detail | Fully grounded in context |

**Correctness** — does it agree with the reference answer?

| 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Contradicts / wrong | Mostly incorrect | Partially correct | Mostly correct, minor errors | Fully correct |

**Answer-relevance** — does it directly address the question asked?

| 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Off-topic | Barely addresses it | Addresses part | Addresses most | Fully on-point |

**Completeness** — does it cover the key points of the reference answer?

| 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Misses key points | Covers little | Covers some | Covers most | Covers all key points |

> The judge prompt currently gives short definitions of each dimension; the
> level anchors above are the intended interpretation. They can be baked into
> the prompt for stricter consistency (requires re-running the judge).

---

## 5. Robustness (stress)

Each question is re-asked under four perturbations — `typo`, `case`, `noise`,
`injection` (a prompt-injection prefix). The system should answer stably.

| Metric | Range | Meaning | Formula |
|---|---|---|---|
| **stability** | 0–1 | Mean semantic similarity between the answer to the original question and the answers to its perturbations. 1.0 = identical/robust; low = the answer changed (or the injection succeeded). | `mean over variants of cosine(embed(base_answer), embed(variant_answer))` |

---

## 6. Normalization → [0,1]

Family scores average their members on a common [0,1] scale:

```
norm_judge_3(x) = x / 3            # chunk relevance (0–3)
norm_judge_5(x) = (x - 1) / 4      # all 1–5 judge scores
bertscore_clamped = max(0.0, bertscore)   # baseline-rescaled BERTScore can be
                                          # negative; clamp before averaging
```

Reference and robustness metrics are already in [0,1].

## 7. Family scores

```
retrieval_ref    = mean(P@k, R@k, MRR, NDCG@k, context_recall)
generation_ref   = mean(ROUGE-L, METEOR, bertscore_clamped)
judge_retrieval  = mean(norm_judge_3(chunk_relevance_avg), norm_judge_5(context_quality))
judge_generation = mean(norm_judge_5(faithfulness), norm_judge_5(correctness),
                        norm_judge_5(answer_relevance), norm_judge_5(completeness))
robustness       = stability
```

## 8. Overall score

A weighted average of the five family scores (weights normalized by their sum):

```
overall = Σ_f ( weight_f · family_f ) / Σ_f weight_f
```

Default weights (`eval/scoring.py:DEFAULT_WEIGHTS`, override with `--weights`):

| Family | Weight | Why this weight |
|---|---|---|
| judge_generation | **0.35** | A strong judge rating correctness/faithfulness is the closest proxy to "is the answer actually good," and is robust to paraphrase — so it carries the most weight. |
| retrieval_ref | **0.20** | Standard IR metrics, but answer-grounded relevance is lenient/inflated → moderate, not dominant. |
| generation_ref | **0.20** | Lexical/semantic overlap with verbose human emails is a weak correctness signal → moderate. |
| judge_retrieval | **0.15** | A valuable *stricter* retrieval signal, but single-model judgment is noisy → below the generation judge. |
| robustness | **0.10** | A stability/safety check; secondary to getting the answer right. |

> These are a **reasonable default heuristic, not empirically tuned.** They
> encode the priority "trust the strong judge's correctness most, treat lexical
> reference metrics as a floor." Adjust to your priorities, e.g.
> `--weights judge_retrieval=0.25,retrieval_ref=0.10`.
