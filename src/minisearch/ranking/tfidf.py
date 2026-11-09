"""TF-IDF ranking, in the style of Lucene's ClassicSimilarity (vector-space model).

    score(q, d) = sum over t in q of  tf_w(t, d) * idf(t)^2 * qtf * norm(d)

    tf_w  = sqrt(tf)                       (or 1 + ln tf with sublinear="log")
    idf   = 1 + ln((N + 1) / (df + 1))
    norm  = 1 / sqrt(|d|)

idf is squared because it appears in both the query vector and the document
vector of the cosine dot product. ``1/sqrt(|d|)`` approximates the
document-vector norm without storing a per-document norm, which would have
to be recomputed whenever IDF changes.

TF never saturates here. That is the main behavioural difference from BM25,
and the relevance evaluation shows its effect.
"""

from __future__ import annotations

import math

import numpy as np

from minisearch.ranking.base import CollectionStats, Ranker, TermScorer


def tfidf_idf(df: int, num_docs: int) -> float:
    df = min(df, num_docs)  # df counts deleted docs until merge
    return 1.0 + math.log((num_docs + 1) / (df + 1))


class TfIdfTermScorer(TermScorer):
    __slots__ = ("weight", "log_tf")

    def __init__(self, idf: float, query_tf: int, log_tf: bool) -> None:
        self.weight = idf * idf * query_tf
        self.log_tf = log_tf

    def _tf(self, tf: float) -> float:
        return 1.0 + math.log(tf) if self.log_tf else math.sqrt(tf)

    def score(self, tf: int, doc_length: int) -> float:
        if tf <= 0:
            return 0.0
        return self.weight * self._tf(tf) / math.sqrt(max(doc_length, 1))

    def score_array(self, tfs: np.ndarray, doc_lengths: np.ndarray) -> np.ndarray:
        tfs = tfs.astype(np.float64, copy=False)
        tfw = 1.0 + np.log(tfs) if self.log_tf else np.sqrt(tfs)
        return self.weight * tfw / np.sqrt(np.maximum(doc_lengths, 1))

    def upper_bound(self, max_tf: int, min_doc_length: int) -> float:
        return self.score(max_tf, min_doc_length)


class TfIdf(Ranker):
    """Vector-space TF-IDF ranker (``sublinear`` = "sqrt" or "log")."""

    name = "tfidf"

    def __init__(self, sublinear: str = "sqrt") -> None:
        if sublinear not in ("sqrt", "log"):
            raise ValueError("sublinear must be 'sqrt' or 'log'")
        self.sublinear = sublinear

    def term_scorer(self, df: int, stats: CollectionStats, query_tf: int = 1) -> TfIdfTermScorer:
        return TfIdfTermScorer(tfidf_idf(df, stats.num_docs), query_tf, self.sublinear == "log")

    def describe(self) -> dict:
        return {"name": self.name, "sublinear": self.sublinear}
