"""
Corpus configuration: which document sources are indexed and how they are
prioritized during retrieval.

The RGD chatbot's primary job is to *guide users around the RGD website*, so
the website help content (``RGDHelpMarkdown/``) is prioritized over the
research papers (``papers/``). Priority is stored on every chunk and used as a
small ranking boost at retrieval time so guidance content wins ties.

Higher ``priority`` = more important. Matching is by path prefix; the longest
(most specific) matching prefix wins, so individual files can override a folder.
"""

from pathlib import PurePath
from typing import Dict

# Default priority when no rule matches.
DEFAULT_PRIORITY = 1

# Per-source priority rules, keyed by path prefix (relative or absolute parts
# are normalized away — we match on the path components).
PRIORITY_RULES: Dict[str, int] = {
    "RGDHelpMarkdown": 5,   # Website guidance — the chatbot's main purpose.
    "papers": 2,            # Background research papers — secondary.
}

# Weight applied per priority point when re-ranking retrieved chunks.
# score = cosine_similarity + PRIORITY_BOOST * priority
# Kept small so it only breaks ties / nudges, never overrides a clearly more
# relevant chunk.
PRIORITY_BOOST = 0.01


def priority_for_source(source: str) -> int:
    """Return the configured priority for a document source path.

    Matches by path component against ``PRIORITY_RULES`` and returns the value
    for the most specific (longest) matching rule.
    """
    parts = set(PurePath(source).parts)
    best_priority = DEFAULT_PRIORITY
    best_specificity = -1
    for prefix, priority in PRIORITY_RULES.items():
        prefix_parts = PurePath(prefix).parts
        # A rule matches if all of its path components appear in the source path.
        if all(p in parts for p in prefix_parts):
            specificity = len(prefix_parts)
            if specificity > best_specificity:
                best_specificity = specificity
                best_priority = priority
    return best_priority
