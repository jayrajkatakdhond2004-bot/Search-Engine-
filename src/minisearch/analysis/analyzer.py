"""Configurable text analysis pipeline.

An :class:`Analyzer` is ``tokenizer -> filter_1 -> ... -> filter_n``. Each
filter maps a term to a new term, or to ``None`` to drop it.

Positions are assigned *before* filtering. A dropped stopword therefore leaves
a gap: in "university of london", ``university`` is at 0 and ``london`` at 2.
The query side is analysed the same way, so phrase matching compares
*relative* offsets and stays correct even though stopwords are not indexed.

The analyzer configuration is stored in the index manifest, so queries are
always analysed exactly like the documents they are matched against.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass

from minisearch.analysis.porter import PorterStemmer
from minisearch.analysis.stopwords import STOPWORD_SETS

TermFilter = Callable[[str], str | None]

# A token is a run of letters/digits, optionally joined by apostrophes ("don't").
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


@dataclass(frozen=True, slots=True)
class Token:
    """A token with its position (token index) and character offsets in the source."""

    term: str
    position: int
    start: int
    end: int


def _fold_accents(term: str) -> str:
    if term.isascii():
        return term
    decomposed = unicodedata.normalize("NFKD", term)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _strip_possessive(term: str) -> str:
    term = term.replace("’", "'")
    if term.endswith("'s"):
        term = term[:-2]
    return term.replace("'", "")


@dataclass(frozen=True)
class AnalyzerConfig:
    """Serializable analyzer settings (persisted in the index manifest)."""

    lowercase: bool = True
    fold_accents: bool = True
    stopwords: str = "lucene"  # one of STOPWORD_SETS
    stemmer: str | None = "porter"  # "porter" or None
    min_length: int = 1
    max_length: int = 40

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AnalyzerConfig:
        return cls(**data)


class Analyzer:
    """Tokenizer followed by a chain of term filters.

    Two entry points share the same logic:

    * :meth:`analyze` returns ``(term, position)`` pairs. This is the fast path
      used for indexing and queries.
    * :meth:`tokens` also yields character offsets, for snippet highlighting.
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()
        self._filters: Sequence[TermFilter] = self._build_filters(self.config)

    @staticmethod
    def _build_filters(cfg: AnalyzerConfig) -> list[TermFilter]:
        filters: list[TermFilter] = []
        if cfg.lowercase:
            filters.append(str.lower)
        if cfg.fold_accents:
            filters.append(_fold_accents)
        filters.append(_strip_possessive)

        stop = STOPWORD_SETS[cfg.stopwords]
        lo, hi = cfg.min_length, cfg.max_length

        def length_and_stop(term: str) -> str | None:
            if len(term) < lo or len(term) > hi or term in stop:
                return None
            return term

        filters.append(length_and_stop)
        if cfg.stemmer == "porter":
            filters.append(PorterStemmer())
        elif cfg.stemmer is not None:
            raise ValueError(f"unknown stemmer {cfg.stemmer!r}")
        return filters

    def normalize(self, term: str) -> str | None:
        """Run a single raw term through the filter chain."""
        result: str | None = term
        for f in self._filters:
            result = f(result)
            if not result:
                return None
        return result

    def analyze(self, text: str) -> list[tuple[str, int]]:
        """Return ``(term, position)`` pairs for ``text``."""
        out: list[tuple[str, int]] = []
        normalize = self.normalize
        for pos, raw in enumerate(_TOKEN_RE.findall(text)):
            term = normalize(raw)
            if term is not None:
                out.append((term, pos))
        return out

    def terms(self, text: str) -> list[str]:
        """Return just the analysed terms of ``text``."""
        return [t for t, _ in self.analyze(text)]

    def tokens(self, text: str) -> Iterator[Token]:
        """Yield tokens with character offsets (used for highlighting)."""
        for pos, m in enumerate(_TOKEN_RE.finditer(text)):
            term = self.normalize(m.group())
            if term is not None:
                yield Token(term, pos, m.start(), m.end())

    def raw_tokens(self, text: str) -> list[str]:
        """Tokenize without filtering (debugging and tests)."""
        return _TOKEN_RE.findall(text)
