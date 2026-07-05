"""
token_engine.py — Stage 1: Pure structural tokenization.

No semantic processing occurs here.
Input:  raw text string
Output: TokenizedDocument with full position/sentence/paragraph metadata

The Unicode regex handles all scripts (Latin, Arabic, Persian, CJK, etc.)
without any language-specific rules. Contractions like "don't" are kept intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from collections import Counter

if TYPE_CHECKING:
    from .feature_extractor import StructuralFeatureExtractor


# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------

# Paragraph boundary: one or more blank lines
_PARA_RE = re.compile(r"\n\s*\n+", re.UNICODE)

# Sentence boundary: end-of-sentence punctuation followed by whitespace, OR newline
# Covers ASCII (.!?) and common Unicode punctuation (؟ ！ ？ ‥ …)
_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?؟！？‥…])\s+|\n",
    re.UNICODE,
)

# Token pattern: Unicode word characters, with optional apostrophe-contraction
_TOKEN_RE = re.compile(r"[\w]+(?:'[\w]+)?", re.UNICODE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class Token:
    """
    A single tokenized unit with full structural metadata.
    text        : normalized (lowercased) form used for graph lookups
    raw         : original form before normalization
    position    : absolute 0-indexed position in document
    sentence_id : which sentence this token belongs to (0-indexed)
    paragraph_id: which paragraph this token belongs to (0-indexed)
    local_window: list of neighboring token texts within window_size
    """
    text:         str
    raw:          str
    position:     int
    sentence_id:  int
    paragraph_id: int
    local_window: List[str] = field(default_factory=list)


@dataclass
class TokenizedDocument:
    """
    Complete tokenized representation of one document.
    tokens     : flat list of all tokens in document order
    sentences  : tokens grouped by sentence_id (list of lists)
    paragraphs : tokens grouped by paragraph_id (list of lists)
    raw_text   : original input string
    doc_id     : optional identifier for logging / debugging
    freq       : token frequency counter (derived from tokens)
    """
    tokens:     List[Token]
    sentences:  List[List[Token]]
    paragraphs: List[List[Token]]
    raw_text:   str
    doc_id:     Optional[str] = None
    freq:       Counter = field(default_factory=Counter)

    @property
    def token_texts(self) -> List[str]:
        return [t.text for t in self.tokens]

    @property
    def unique_texts(self) -> List[str]:
        return list({t.text for t in self.tokens})

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)


# ---------------------------------------------------------------------------
# TokenEngine
# ---------------------------------------------------------------------------

class TokenEngine:
    """
    Stateless tokenizer. Produces TokenizedDocument from raw text.

    Parameters
    ----------
    window_size      : int
        Radius of local context window filled for each token.
        Token at position i gets neighbors from [i-window_size, i+window_size].
    feature_extractor: StructuralFeatureExtractor | None
        Optional pre-tokenization module. When set:
          - text is HTML-decoded before tokenization via extractor.preprocess()
          - structural pseudo-tokens (e.g. __LONG_NUM__, __CAPS_WORD__) are
            appended at the end of every TokenizedDocument via extractor.extract()
        These pseudo-tokens flow through the RelationGraph pipeline as regular nodes.
    """

    def __init__(
        self,
        window_size: int = 5,
        feature_extractor: "Optional[StructuralFeatureExtractor]" = None,
    ) -> None:
        self.window_size = window_size
        self.feature_extractor = feature_extractor

    def tokenize(
        self,
        text: str,
        doc_id: Optional[str] = None,
    ) -> TokenizedDocument:
        """
        Tokenize raw text into a TokenizedDocument.

        Algorithm
        ---------
        1. Split into paragraphs on double-newline.
        2. Split each paragraph into sentence chunks.
        3. Apply Unicode regex to extract tokens from each sentence.
        4. Record absolute position, sentence_id, paragraph_id for every token.
        5. Second pass: fill local_window for all tokens.
        6. Build sentence/paragraph groupings and frequency counter.
        """
        # Pre-processing: HTML decode + whitespace normalization
        if self.feature_extractor:
            text = self.feature_extractor.preprocess(text)

        if not text:
            return TokenizedDocument(
                tokens=[], sentences=[], paragraphs=[],
                raw_text=text or "", doc_id=doc_id,
            )

        tokens: List[Token] = []
        global_token_pos  = 0
        global_sentence_id = 0

        # Split into paragraphs (keep non-empty chunks)
        para_chunks = [p.strip() for p in _PARA_RE.split(text) if p and p.strip()]
        if not para_chunks:
            para_chunks = [text.strip()]

        for para_id, para_chunk in enumerate(para_chunks):
            # Split paragraph into sentence chunks
            sent_chunks = [s.strip() for s in _SENT_SPLIT_RE.split(para_chunk)
                           if s and s.strip()]
            if not sent_chunks:
                sent_chunks = [para_chunk]

            for local_sent_id, sent_str in enumerate(sent_chunks):
                sent_id = global_sentence_id + local_sent_id
                raw_toks = _TOKEN_RE.findall(sent_str)

                for raw_tok in raw_toks:
                    tokens.append(Token(
                        text=raw_tok.lower(),
                        raw=raw_tok,
                        position=global_token_pos,
                        sentence_id=sent_id,
                        paragraph_id=para_id,
                        local_window=[],
                    ))
                    global_token_pos += 1

            global_sentence_id += max(len(sent_chunks), 1)

        if not tokens:
            return TokenizedDocument(
                tokens=[], sentences=[], paragraphs=[],
                raw_text=text, doc_id=doc_id,
            )

        # Append structural pseudo-tokens (from FeatureExtractor) as extra tokens.
        # They get positions after all normal tokens; sentence/paragraph IDs are 0.
        # They participate in RelationGraph update and direct evidence normally.
        if self.feature_extractor and tokens:
            base_pos = len(tokens)
            structural = self.feature_extractor.extract(text)
            # Word-bigram pseudo-tokens: extracted from normal tokens only.
            # These provide phrase-level disambiguation (free_entry vs free_time).
            normal_words = [t.text for t in tokens]
            bigrams = self.feature_extractor.extract_bigrams(normal_words)
            extra_tokens = structural + bigrams
            for i, feat_text in enumerate(extra_tokens):
                tokens.append(Token(
                    text=feat_text,
                    raw=feat_text,
                    position=base_pos + i,
                    sentence_id=0,
                    paragraph_id=0,
                    local_window=[],
                ))

        # --- Second pass: fill local_window ---
        n = len(tokens)
        W = self.window_size
        for i, tok in enumerate(tokens):
            left  = max(0, i - W)
            right = min(n, i + W + 1)
            tok.local_window = [
                tokens[j].text for j in range(left, right) if j != i
            ]

        # --- Group by sentence_id ---
        sent_map: Dict[int, List[Token]] = {}
        for tok in tokens:
            sent_map.setdefault(tok.sentence_id, []).append(tok)
        sentences = [sent_map[k] for k in sorted(sent_map.keys())]

        # --- Group by paragraph_id ---
        para_map: Dict[int, List[Token]] = {}
        for tok in tokens:
            para_map.setdefault(tok.paragraph_id, []).append(tok)
        paragraphs = [para_map[k] for k in sorted(para_map.keys())]

        # --- Frequency counter ---
        freq = Counter(tok.text for tok in tokens)

        return TokenizedDocument(
            tokens=tokens,
            sentences=sentences,
            paragraphs=paragraphs,
            raw_text=text,
            doc_id=doc_id,
            freq=freq,
        )

    def tokenize_batch(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> List[TokenizedDocument]:
        """Tokenize a list of texts. Returns list of TokenizedDocuments."""
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(texts))]
        return [self.tokenize(text, doc_id=did) for text, did in zip(texts, doc_ids)]
