"""
Parse the RGD help-desk export (qa_report.html), clean reference answers, and
select a fixed, answerable evaluation set.

The same selected set is cached to ``data/eval/eval_set.jsonl`` so every metric
in the harness runs on identical questions across runs.
"""

import json
import os
import random
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

# Question types worth evaluating as "website guidance" (vs curation/admin).
ANSWERABLE_TYPES = {"Help", "Tool"}

# Question/answer patterns that signal an administrative or curation request
# rather than a guidance question the corpus could answer.
_ADMIN_PATTERNS = [
    r"\bupdate (our|the|my|your)\b",
    r"\bwe have updated\b",
    r"\bhas been (updated|fixed|added|removed)\b",
    r"\bbroken link\b",
    r"\bplease (add|remove|update|delete|change)\b",
    r"\bunsubscribe\b",
    r"\bremove (me|our|my)\b",
    r"\bnomenclature (change|update|request)\b",
    r"\bshould be fixed\b",
]
_ADMIN_RE = re.compile("|".join(_ADMIN_PATTERNS), re.IGNORECASE)

# Signature / greeting markers used to trim real-email reference answers.
_SIGNATURE_MARKERS = [
    "sincerely", "best regards", "kind regards", "warm regards", "regards,",
    "best,", "cheers", "thanks,", "thank you,", "many thanks",
]
_GREETING_RE = re.compile(r"^\s*(hello|hi|dear|greetings|good (morning|afternoon))\b.*$", re.IGNORECASE)
_CONTACT_RE = re.compile(
    r"(medical college of wisconsin|rat genome database|watertown plank|"
    r"\bph:\s|\bfax:\s|@\w+\.\w+|http[s]?://|\b\d{3}-\d{3}-\d{4}\b)",
    re.IGNORECASE,
)

# Promotional / spam patterns (marketing, SEO, link-building, solicitations).
_SPAM_RE = re.compile(
    r"(professional (writ|seo|market)|guest post|back ?link|link building|"
    r"we (are|represent) one of|collaborat\w* opportunit|promote your|"
    r"digital marketing|increase your (traffic|ranking)|business proposal|"
    r"addiction (rehab|treatment|cent)|rehabilitation cent|"
    r"\bcasino\b|\bcrypto\b|investment opportunit|sponsorship)",
    re.IGNORECASE,
)

# Refusal / out-of-scope answers — the human reply declined rather than guiding
# the user through the website, so it is not a useful generation reference.
_DECLINE_RE = re.compile(
    r"(not medical (doctors|professionals)|cannot give .*medical advice|"
    r"we are not able to|we are unable to|i'?m afraid (we|that)|"
    r"unfortunately,? we (cannot|can't|do not|don'?t)|we do not provide|"
    r"outside (the |our )?scope|consult (your|a) (physician|doctor|genetic))",
    re.IGNORECASE,
)

# Interrogative cues used to confirm the message is an actual question.
_INTERROGATIVE_RE = re.compile(
    r"(\?|\b(how|what|where|which|when|why|who|can|could|do|does|is|are|"
    r"should|would|i'?d like to know|looking for|trying to)\b)",
    re.IGNORECASE,
)


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_ascii = sum(1 for c in text if ord(c) > 127 or c == "�")
    return non_ascii / len(text)


def _is_spam(question: str) -> bool:
    return bool(_SPAM_RE.search(question))


def _is_decline(answer: str) -> bool:
    return bool(_DECLINE_RE.search(answer))


def _looks_like_question(question: str) -> bool:
    return bool(_INTERROGATIVE_RE.search(question))


# Cues that a question is about *using/navigating the RGD website* (web-guide)
# rather than a raw data/research request.
_GUIDE_RE = re.compile(
    r"(how (do|can|would|to)|where (can|do|is|are|to|should)|"
    r"\bfind\b|\bsearch\b|\blook ?up\b|\blocate\b|\bdownload\b|\bexport\b|\baccess\b|"
    r"\bview\b|\bdisplay\b|\bnavigate\b|\bbrowse\b|\bfilter\b|"
    r"\bpage\b|\btool\b|\bbrowser\b|jbrowse|genome browser|ontology browser|"
    r"\bportal\b|\btrack\b|report page|\bwebsite\b|\bweb ?site\b|\bsite\b|"
    r"\blink\b|\bmenu\b|\btab\b|how is .* (used|displayed)|what does .* mean)",
    re.IGNORECASE,
)


def _guide_score(question: str) -> int:
    """Count distinct web-guide cues in a question (higher = more how-to)."""
    return len(set(m.group(0).lower() for m in _GUIDE_RE.finditer(question)))


# Strong web-guide phrasing: an explicit how-to / where-to question about using
# the site. This is the gate (the cue count above is just supplementary).
_STRONG_GUIDE_RE = re.compile(
    r"(how (do|can|could|would|should|to)\b|"
    r"where (can|do|to|should|is|are)\b|"
    r"how (is|are|do|can) .{0,50}\b(use|used|using|display|view|access|download|find|search)\b|"
    r"what does .{0,50}\bmean\b|how is .{0,40}\b(calculated|defined|displayed))",
    re.IGNORECASE,
)


def _is_strong_guide(question: str) -> bool:
    return bool(_STRONG_GUIDE_RE.search(question))


def _block_text(node) -> str:
    """Extract a div's text, converting <br> to newlines."""
    for br in node.find_all("br"):
        br.replace_with("\n")
    return node.get_text().strip()


def parse_qa_report(path: str) -> List[Dict]:
    """Parse all question cards from the HTML export.

    Returns a list of dicts: id, subject, type, date, question, answer, answered.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    items = []
    for card in soup.select("div.question-card"):
        classes = card.get("class", [])
        answered = "answered" in classes and "unanswered" not in classes

        qid = _first_text(card.select_one(".q-id")).lstrip("#")
        subject = _first_text(card.select_one(".q-subject"))
        qtype = card.get("data-type", "") or _first_text(card.select_one(".q-badge"))
        date = _first_text(card.select_one(".q-date"))

        msg = card.select_one(".message-block")
        question = _block_text(msg) if msg else ""

        ans_node = card.select_one(".answer-block .answer-text")
        answer = _block_text(ans_node) if ans_node else ""

        items.append({
            "id": qid,
            "subject": subject,
            "type": qtype,
            "date": date,
            "question": question,
            "answer": answer,
            "answered": answered,
        })
    return items


def _first_text(node) -> str:
    return node.get_text().strip() if node else ""


def clean_answer(text: str) -> str:
    """Strip greeting and signature/contact blocks from a help-desk answer."""
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]

    # Drop leading greeting line(s).
    while lines and (not lines[0].strip() or _GREETING_RE.match(lines[0])):
        lines.pop(0)

    # Cut at the first signature marker line.
    cut = len(lines)
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if any(low.startswith(m) or low == m.rstrip(",") for m in _SIGNATURE_MARKERS):
            cut = i
            break
    lines = lines[:cut]

    # Drop trailing contact-info lines (signature block without a marker).
    while lines and (_CONTACT_RE.search(lines[-1]) or not lines[-1].strip()):
        lines.pop()

    return "\n".join(ln for ln in lines if ln.strip()).strip()


def _is_admin(item: Dict) -> bool:
    return bool(_ADMIN_RE.search(item["question"]) or _ADMIN_RE.search(item["answer"]))


def filter_candidates(
    items: List[Dict],
    min_question_chars: int = 20,
    min_answer_chars: int = 40,
    max_non_ascii_ratio: float = 0.05,
) -> List[Dict]:
    """Apply all content filters (no corpus-coverage gate) and return the
    surviving answered Help/Tool Q&A, each with a cleaned ``reference_answer``."""
    candidates = []
    for it in items:
        if not it["answered"] or it["type"] not in ANSWERABLE_TYPES:
            continue
        if _is_admin(it) or _is_spam(it["question"]):
            continue
        if len(it["question"]) < min_question_chars:
            continue
        # Drop garbled / non-English messages (encoding noise, foreign scripts).
        if _non_ascii_ratio(it["question"]) > max_non_ascii_ratio:
            continue
        if not _looks_like_question(it["question"]):
            continue
        cleaned = clean_answer(it["answer"])
        if len(cleaned) < min_answer_chars:
            continue
        # Drop refusal / out-of-scope answers — not corpus-grounded guidance.
        if _is_decline(cleaned):
            continue
        candidates.append({**it, "reference_answer": cleaned})
    return candidates


def select_eval_set(
    items: List[Dict],
    rag,
    n: int = 50,
    seed: int = 0,
    coverage_threshold: float = 0.5,
    min_question_chars: int = 20,
    min_answer_chars: int = 40,
    max_non_ascii_ratio: float = 0.05,
    require_guide: bool = True,
) -> List[Dict]:
    """Filter to answerable guidance questions and deterministically sample ``n``.

    Filtering keeps answered Help/Tool questions, drops administrative/curation
    requests, and (when ``require_guide``) keeps only **web-guide / how-to**
    questions about using the RGD website. Gates on corpus coverage (top-1
    retrieval similarity). Each kept item gets a cleaned ``reference_answer``.
    """
    candidates = filter_candidates(
        items, min_question_chars=min_question_chars,
        min_answer_chars=min_answer_chars, max_non_ascii_ratio=max_non_ascii_ratio,
    )

    # Keep web-guide / how-to questions (explicit how-to/where phrasing).
    if require_guide:
        for c in candidates:
            c["guide_score"] = _guide_score(c["question"])
        candidates = [c for c in candidates if _is_strong_guide(c["question"])]

    # Deterministic order before the coverage gate.
    candidates.sort(key=lambda x: x["id"])
    random.Random(seed).shuffle(candidates)

    selected = []
    for it in candidates:
        retrieved = rag.retrieve(it["question"], n_results=1, min_similarity=0.0)
        top_sim = retrieved[0]["similarity"] if retrieved else 0.0
        if top_sim >= coverage_threshold:
            it["top1_similarity"] = round(float(top_sim), 4)
            selected.append(it)
        if len(selected) >= n:
            break
    return selected


def load_or_build_eval_set(
    qa_report: str,
    rag,
    out_path: str,
    n: int = 50,
    seed: int = 0,
    rebuild: bool = False,
) -> List[Dict]:
    """Return the cached eval set, building and caching it if needed."""
    if os.path.exists(out_path) and not rebuild:
        with open(out_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    items = parse_qa_report(qa_report)
    selected = select_eval_set(items, rag, n=n, seed=seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in selected:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return selected


def dump_all_qa(qa_report: str, out_path: str) -> List[Dict]:
    """Write every parsed Q&A pair from the report to JSONL (no filtering).

    Each record carries the raw fields plus ``reference_answer`` (the cleaned
    answer). Useful as a full dataset alongside the filtered ``eval_set.jsonl``.
    """
    items = parse_qa_report(qa_report)
    for it in items:
        it["reference_answer"] = clean_answer(it["answer"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


def dump_candidates(qa_report: str, out_path: str) -> List[Dict]:
    """Write the content-filtered candidate Q&A (pre coverage-gate) to JSONL."""
    candidates = filter_candidates(parse_qa_report(qa_report))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in candidates:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return candidates


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Export website Q&A to JSONL")
    p.add_argument("--qa-report", default="qa_report.html")
    p.add_argument("--out", default="data/eval/all_qa.jsonl", help="Output for --all")
    p.add_argument("--candidates-out", default=None,
                   help="If set, also write the content-filtered candidate Q&A here")
    p.add_argument("--all", action="store_true", help="Dump all parsed Q&A (default)")
    a = p.parse_args()

    if a.candidates_out:
        cands = dump_candidates(a.qa_report, a.candidates_out)
        print(f"Wrote {len(cands)} filtered candidate Q&A to {a.candidates_out}")
    if a.all or not a.candidates_out:
        items = dump_all_qa(a.qa_report, a.out)
        answered = sum(1 for i in items if i["answered"])
        print(f"Wrote {len(items)} Q&A ({answered} answered) to {a.out}")
