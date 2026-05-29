"""
Sentence-aware document chunking.

The previous approach sliced text every N characters, which routinely split
sentences (and even words) mid-stream and hurt retrieval quality. This module
replaces that with structure- and sentence-aware chunking, mirroring the
ideas in the SCGE Spring AI reference (TokenTextSplitter + a quality filter):

- Markdown is split on its header hierarchy first (so a chunk stays within one
  section and carries the section title as metadata), then size-capped.
- PDFs / plain text are split on paragraph and sentence boundaries.
- Chunk size is measured in tokens when ``tiktoken`` is available, otherwise a
  character heuristic is used.
- Low-quality chunks (too short, or mostly formatting characters) are dropped.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Target ~800 tokens per chunk with ~150 token overlap, matching the reference.
CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 150
# Rough chars-per-token for English prose, used for sizing and the char fallback.
CHARS_PER_TOKEN = 4

# Separators tried in order — paragraph, line, sentence, clause, space, char.
# Splitting prefers the earliest separator, so breaks land on sentence/paragraph
# boundaries before ever falling back to mid-word.
SENTENCE_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

# Markdown header levels to split on, mapped to the metadata key that records them.
MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Quality-filter thresholds (ported from DocumentPreprocessor.isQualityChunk).
MIN_CHUNK_CHARS = 50
MAX_FORMATTING_RATIO = 0.6
_FORMATTING_CHARS = set("|-_=*+~^<>[]{}()")


@dataclass
class Chunk:
    """A single chunk of a document, ready for embedding."""

    content: str
    metadata: Dict[str, object] = field(default_factory=dict)


def _token_len(text: str) -> int:
    """Length of ``text`` in tokens (tiktoken if available, else char heuristic)."""
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // CHARS_PER_TOKEN)


_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    """Lazily load a tiktoken encoder; return None if tiktoken isn't installed."""
    global _ENCODER, _ENCODER_TRIED
    if not _ENCODER_TRIED:
        _ENCODER_TRIED = True
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


def _build_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """Recursive splitter sized in tokens, breaking on sentence boundaries."""
    enc = _get_encoder()
    if enc is not None:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=CHUNK_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            separators=SENTENCE_SEPARATORS,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS * CHARS_PER_TOKEN,
        chunk_overlap=CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN,
        separators=SENTENCE_SEPARATORS,
        length_function=len,
    )


def is_quality_chunk(text: str) -> bool:
    """Reject chunks that are too short or mostly formatting noise."""
    stripped = text.strip()
    if len(stripped) < MIN_CHUNK_CHARS:
        return False
    formatting = sum(1 for c in stripped if c in _FORMATTING_CHARS)
    if formatting / len(stripped) > MAX_FORMATTING_RATIO:
        return False
    # Require at least a few "real" words (not just numbers/punctuation).
    meaningful = [w for w in stripped.split() if len(w) > 1 and not w.isdigit()]
    return len(meaningful) >= 5


def _section_label(header_meta: Dict[str, str]) -> str:
    """Join captured header levels into a single readable section path."""
    parts = [header_meta[key] for _, key in MARKDOWN_HEADERS if key in header_meta]
    return " > ".join(parts)


def chunk_document(
    text: str,
    file_type: str,
    source: str,
    base_metadata: Optional[Dict[str, object]] = None,
) -> List[Chunk]:
    """Split ``text`` into sentence-aware chunks.

    Args:
        text: Full document text.
        file_type: One of ``"markdown"``/``"md"``, ``"pdf"``, ``"text"``.
        source: Source path, recorded on every chunk.
        base_metadata: Extra metadata copied onto every chunk.

    Returns:
        List of quality-filtered :class:`Chunk` objects.
    """
    base = dict(base_metadata or {})
    base.setdefault("source", source)

    recursive = _build_recursive_splitter()
    chunks: List[Chunk] = []

    is_markdown = file_type.lower() in {"markdown", "md"} or source.lower().endswith(".md")

    if is_markdown:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS,
            strip_headers=False,
        )
        sections = header_splitter.split_text(text)
        for section in sections:
            section_label = _section_label(section.metadata)
            # Size-cap each section so long sections become multiple chunks,
            # still on sentence boundaries.
            for piece in recursive.split_text(section.page_content):
                if not is_quality_chunk(piece):
                    continue
                meta = dict(base)
                if section_label:
                    meta["section"] = section_label
                chunks.append(Chunk(content=piece, metadata=meta))
    else:
        for piece in recursive.split_text(text):
            if not is_quality_chunk(piece):
                continue
            chunks.append(Chunk(content=piece, metadata=dict(base)))

    # Record positional index per source for stable, ordered chunk ids.
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    return chunks
