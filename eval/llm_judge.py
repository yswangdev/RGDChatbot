"""
LLM-as-judge scoring using a model stronger than the generator (qwen2.5:7b) to
reduce self-grading bias.

Retrieval: per-chunk relevance (0-3) and overall context sufficiency (1-5).
Generation: faithfulness, correctness, answer-relevance, completeness (1-5).
All prompts request strict JSON; responses are parsed defensively.
"""

import json
import re
from typing import Dict, List

import ollama


def _extract_json(text: str) -> Dict:
    """Pull the first JSON object out of a model response."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


class LLMJudge:
    def __init__(self, host: str, model: str = "qwen2.5:7b"):
        self.client = ollama.Client(host=host)
        self.model = model

    def _ask(self, system: str, user: str) -> Dict:
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format="json",
                options={"temperature": 0.0},
            )
            return _extract_json(resp["message"]["content"])
        except Exception as e:
            print(f"  (judge error: {e})")
            return {}

    def judge_chunk_relevance(self, question: str, chunk: str) -> int:
        """Score how relevant a single chunk is to the question (0-3)."""
        system = (
            "You are evaluating a retrieval system. Rate how relevant the passage is "
            "to answering the question on a 0-3 scale: 0=irrelevant, 1=slightly, "
            "2=relevant, 3=highly relevant. Respond ONLY as JSON: {\"score\": <int>}."
        )
        user = f"QUESTION:\n{question}\n\nPASSAGE:\n{chunk[:1500]}"
        out = self._ask(system, user)
        return int(_clamp(out.get("score"), 0, 3, 0))

    def judge_context_quality(self, question: str, contexts: List[str]) -> int:
        """Rate whether the retrieved context as a whole is sufficient (1-5)."""
        system = (
            "You evaluate retrieval quality. Given a question and the retrieved passages, "
            "rate how sufficient and on-topic the passages are for answering the question "
            "on a 1-5 scale (1=useless, 5=fully sufficient). "
            "Respond ONLY as JSON: {\"score\": <int>}."
        )
        joined = "\n\n---\n\n".join(c[:1000] for c in contexts)
        user = f"QUESTION:\n{question}\n\nRETRIEVED PASSAGES:\n{joined}"
        out = self._ask(system, user)
        return int(_clamp(out.get("score"), 1, 5, 1))

    def judge_generation(self, question: str, context: str, candidate: str, reference: str) -> Dict[str, float]:
        """Rate the generated answer on four 1-5 dimensions."""
        system = (
            "You are a strict evaluator of a question-answering system. Rate the candidate "
            "answer on four dimensions, each 1-5:\n"
            "- faithfulness: is every claim supported by the CONTEXT (no hallucination)?\n"
            "- correctness: does it agree with the REFERENCE answer?\n"
            "- answer_relevance: does it directly address the QUESTION?\n"
            "- completeness: does it cover the key points of the REFERENCE answer?\n"
            "Respond ONLY as JSON: {\"faithfulness\":<int>,\"correctness\":<int>,"
            "\"answer_relevance\":<int>,\"completeness\":<int>,\"rationale\":\"<short>\"}."
        )
        user = (
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context[:3000]}\n\n"
            f"CANDIDATE ANSWER:\n{candidate}\n\nREFERENCE ANSWER:\n{reference}"
        )
        out = self._ask(system, user)
        return {
            "faithfulness": _clamp(out.get("faithfulness"), 1, 5, 1),
            "correctness": _clamp(out.get("correctness"), 1, 5, 1),
            "answer_relevance": _clamp(out.get("answer_relevance"), 1, 5, 1),
            "completeness": _clamp(out.get("completeness"), 1, 5, 1),
            "rationale": str(out.get("rationale", ""))[:300],
        }
