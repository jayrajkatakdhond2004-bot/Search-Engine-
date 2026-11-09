"""Okapi BM25.

    score(q, d) = sum over t in q of  IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

    IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

* ``k1`` controls TF saturation. A term's contribution approaches
  ``IDF * (k1 + 1)`` however often it occurs. Plain TF-IDF does not saturate,
  so keyword stuffing pays off there.
* ``b`` controls length normalisation. ``b=0`` ignores length, ``b=1`` fully
  normalises by ``|d| / avgdl``.
* The ``1 +`` inside the log is the Lucene variant. It keeps IDF positive for
  terms that occur in more than half the documents. The original
  Robertson-Sparck Jones IDF goes negative there, which breaks the "more
  matches never hurt" property that WAND relies on.
"""

from __future__ import annotations

import math

import numpy as np

from minisearch.ranking.base import CollectionStats, Ranker, TermScorer


def bm25_idf(df: int, num_docs: int) -> float:
    """BM25 inverse document frequency (always > 0)."""
    return math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))


class BM25TermScorer(TermScorer):
    __slots__ = ("weight", "k1", "b", "avgdl", "_k1_1mb", "_k1_b_over_avgdl")

    def __init__(self, idf: float, k1: float, b: float, avgdl: float, query_tf: int) -> None:
        self.weight = idf * (k1 + 1.0) * query_tf
        self.k1, self.b, self.avgdl = k1, b, avgdl
        # Precompute K = k1 * (1 - b + b*dl/avgdl) = k1(1-b) + (k1*b/avgdl) * dl
        self._k1_1mb = k1 * (1.0 - b)
        self._k1_b_over_avgdl = k1 * b / avgdl if avgdl > 0 else 0.0

    def score(self, tf: int, doc_length: int) -> float:
        return self.weight * tf / (tf + self._k1_1mb + self._k1_b_over_avgdl * doc_length)

    def score_array(self, tfs: np.ndarray, doc_lengths: np.ndarray) -> np.ndarray:
        tfs = tfs.astype(np.float64, copy=False)
        return self.weight * tfs / (tfs + self._k1_1mb + self._k1_b_over_avgdl * doc_lengths)

    def upper_bound(self, max_tf: int, min_doc_length: int) -> float:
        return self.score(max_tf, min_doc_length)


class BM25(Ranker):
    """Okapi BM25 with configurable ``k1`` (TF saturation) and ``b`` (length normalisation)."""

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b

    def term_scorer(self, df: int, stats: CollectionStats, query_tf: int = 1) -> BM25TermScorer:
        return BM25TermScorer(bm25_idf(df, stats.num_docs), self.k1, self.b, stats.avgdl, query_tf)

    def describe(self) -> dict:
        return {"name": self.name, "k1": self.k1, "b": self.b}
