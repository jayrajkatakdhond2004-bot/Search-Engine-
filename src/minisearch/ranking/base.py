"""Pluggable ranking functions.

A :class:`Ranker` turns a query term's collection statistics into a
:class:`TermScorer`. The scorer then scores ``(tf, doc_length)`` pairs.
Splitting it this way computes the per-term constants (IDF, normalisation)
once per query term, not once per posting.

Every scorer can also give an ``upper_bound``: the largest score any posting
of the term can get. The WAND top-K algorithm uses it to skip documents
that cannot enter the heap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CollectionStats:
    """Global statistics. They are aggregated over *all* segments, never per segment."""

    num_docs: int
    total_length: int

    @property
    def avgdl(self) -> float:
        return self.total_length / self.num_docs if self.num_docs else 0.0


class TermScorer(ABC):
    """Scores postings of one query term."""

    @abstractmethod
    def score(self, tf: int, doc_length: int) -> float:
        """Score a single posting."""

    @abstractmethod
    def score_array(self, tfs: np.ndarray, doc_lengths: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`score` over whole postings blocks."""

    @abstractmethod
    def upper_bound(self, max_tf: int, min_doc_length: int) -> float:
        """Bound on :meth:`score` over postings with ``tf <= max_tf`` and ``dl >= min_doc_length``.

        Valid because every supported scorer is non-decreasing in ``tf`` and
        non-increasing in document length.
        """


class Ranker(ABC):
    """Factory of per-term scorers. Implementations must be stateless and picklable."""

    name: str = "ranker"

    @abstractmethod
    def term_scorer(self, df: int, stats: CollectionStats, query_tf: int = 1) -> TermScorer:
        """Build a scorer for a term with document frequency ``df``."""

    def describe(self) -> dict:
        return {"name": self.name}
