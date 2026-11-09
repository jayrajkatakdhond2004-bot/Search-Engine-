"""Multi-segment searcher.

Collection statistics (N, avgdl, df) are summed over all segments before
scoring. Scoring each segment with its own statistics is a classic bug: a
segment holding 10 docs gives very different IDFs from one holding 1M, so
scores would not be comparable across segments and results would depend on
merge timing.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from minisearch.analysis import Analyzer
from minisearch.indexer.segment import Segment
from minisearch.ranking import BM25, CollectionStats, Ranker


@dataclass(slots=True)
class Hit:
    doc_id: int
    score: float
    fields: dict = field(default_factory=dict)


@dataclass(slots=True)
class SearchResults:
    hits: list[Hit]
    total_hits: int
    took_ms: float
    query_terms: list[str] = field(default_factory=list)


class IndexSearcher:
    """Searches a point-in-time snapshot of segments."""

    def __init__(self, segments: Sequence[Segment], analyzer: Analyzer) -> None:
        self.segments = list(segments)
        self.analyzer = analyzer
        self.max_doc = max((s.doc_base + s.max_doc for s in self.segments), default=0)
        self._stats: CollectionStats | None = None
        self._live_mask: np.ndarray | None = None

    # ------------------------------------------------------------ statistics

    @property
    def stats(self) -> CollectionStats:
        if self._stats is None:
            self._stats = CollectionStats(
                num_docs=sum(s.num_live_docs for s in self.segments),
                total_length=sum(s.total_length for s in self.segments),
            )
        return self._stats

    def doc_freq(self, term: str) -> int:
        """Global document frequency.

        Deleted docs are still counted until their segment is merged, as in
        Lucene. The approximation is cheap, and merging removes it.
        """
        df = 0
        for seg in self.segments:
            info = seg.term_info(term)
            if info is not None:
                df += info.df
        return df

    @property
    def live_mask(self) -> np.ndarray:
        """Boolean array over global doc IDs: True where the doc is live."""
        if self._live_mask is None:
            mask = np.zeros(self.max_doc, dtype=bool)
            for seg in self.segments:
                mask[seg.doc_base : seg.doc_base + seg.max_doc] = True
                if seg.deleted:
                    mask[np.fromiter(seg.deleted, dtype=np.int64)] = False
            self._live_mask = mask
        return self._live_mask

    def segment_for(self, doc_id: int) -> Segment:
        for seg in self.segments:
            if seg.contains_doc(doc_id):
                return seg
        raise KeyError(doc_id)

    def stored(self, doc_id: int) -> dict:
        return self.segment_for(doc_id).stored(doc_id)

    # --------------------------------------------------------------- scoring

    def score_terms(
        self, terms: Sequence[str], ranker: Ranker, candidates: np.ndarray | None = None
    ) -> np.ndarray:
        """Term-at-a-time scoring into a dense accumulator over global doc IDs.

        Each term's postings are scored as one numpy expression, which is far
        faster in Python than a per-posting loop. ``candidates`` (bool mask)
        restricts scoring to documents that matched a boolean filter.
        """
        acc = np.zeros(self.max_doc, dtype=np.float64)
        stats = self.stats
        for term, qtf in Counter(terms).items():
            df = self.doc_freq(term)
            if df == 0:
                continue
            scorer = ranker.term_scorer(df, stats, qtf)
            for seg in self.segments:
                p = seg.postings(term)
                if p is None:
                    continue
                dls = seg.doc_lengths[p.doc_ids - seg.doc_base]
                acc[p.doc_ids] += scorer.score_array(p.tfs, dls)
        if candidates is not None:
            acc[~candidates] = 0.0
        acc[~self.live_mask] = 0.0
        return acc

    @staticmethod
    def top_k(acc: np.ndarray, k: int, offset: int = 0) -> tuple[np.ndarray, int]:
        """Top ``offset + k`` doc IDs by score, in O(n + k log k) via argpartition."""
        matched = np.flatnonzero(acc > 0)
        total = len(matched)
        need = min(offset + k, total)
        if need == 0:
            return np.empty(0, dtype=np.int64), total
        scores = acc[matched]
        if need < total:
            part = np.argpartition(-scores, need - 1)[:need]
        else:
            part = np.arange(total)
        # Stable tie-break on doc ID keeps results deterministic.
        order = np.lexsort((matched[part], -scores[part]))
        return matched[part][order][offset : offset + k], total

    # ---------------------------------------------------------------- search

    def search(
        self, text: str, k: int = 10, ranker: Ranker | None = None, offset: int = 0
    ) -> SearchResults:
        """Ranked disjunctive (OR) search over free text."""
        t0 = time.perf_counter()
        ranker = ranker or BM25()
        terms = self.analyzer.terms(text)
        acc = self.score_terms(terms, ranker)
        ids, total = self.top_k(acc, k, offset)
        hits = [Hit(int(d), float(acc[d]), self.stored(int(d))) for d in ids]
        return SearchResults(hits, total, (time.perf_counter() - t0) * 1000, terms)
