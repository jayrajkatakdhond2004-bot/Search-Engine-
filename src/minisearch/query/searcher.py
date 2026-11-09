"""Multi-segment searcher: parse -> normalise -> pick engine -> top-K -> hits.

Collection statistics (N, avgdl, df) are summed over all segments before
scoring. Scoring each segment with its own statistics is a classic bug: a
segment holding 10 docs gives very different IDFs from one holding 1M, so
scores would not be comparable across segments and results would depend on
merge timing.

Two engines execute queries (they return identical rankings; tests check this):

* ``vector``: dense boolean masks + numpy term-at-a-time scoring. Handles
  every query shape.
* ``daat``: document-at-a-time iterators with skip pointers, WAND for OR
  queries, zig-zag intersection for AND queries. It touches far fewer
  postings, but pays CPython's per-posting interpreter cost.

``engine="auto"`` uses the vector engine, which is faster in CPython (see the
benchmarks for the numbers and the reasoning).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from minisearch.analysis import Analyzer
from minisearch.indexer.segment import Postings, Segment
from minisearch.query.parser import Node, parse_query
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
    parsed: str = ""
    engine: str = "vector"
    total_is_exact: bool = True
    stats: dict = field(default_factory=dict)


def query_terms(node: Node | None) -> list[str]:
    """Positive (non-negated) terms of a query, for highlighting."""
    from minisearch.query.vector_engine import collect_units

    units = collect_units(node)
    terms = list(units.terms)
    for ph in units.phrases:
        terms.extend(ph.terms)
    return list(dict.fromkeys(terms))


class IndexSearcher:
    """Searches a point-in-time snapshot of segments."""

    def __init__(self, segments: Sequence[Segment], analyzer: Analyzer) -> None:
        self.segments = sorted(segments, key=lambda s: s.doc_base)
        self.analyzer = analyzer
        self.max_doc = max((s.doc_base + s.max_doc for s in self.segments), default=0)
        self._stats: CollectionStats | None = None
        self._live_mask: np.ndarray | None = None
        self._doc_lengths: np.ndarray | None = None
        self._df_cache: dict[str, int] = {}

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
        df = self._df_cache.get(term)
        if df is None:
            df = 0
            for seg in self.segments:
                info = seg.term_info(term)
                if info is not None:
                    df += info.df
            self._df_cache[term] = df
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

    @property
    def doc_lengths_global(self) -> np.ndarray:
        if self._doc_lengths is None:
            dl = np.zeros(self.max_doc, dtype=np.int64)
            for seg in self.segments:
                dl[seg.doc_base : seg.doc_base + seg.max_doc] = seg.doc_lengths
            self._doc_lengths = dl
        return self._doc_lengths

    def term_postings(self, term: str) -> list[Postings]:
        """The term's postings in each segment that contains it (doc-base order)."""
        out = []
        for seg in self.segments:
            p = seg.postings(term)
            if p is not None and len(p):
                out.append(p)
        return out

    def segment_for(self, doc_id: int) -> Segment:
        for seg in self.segments:
            if seg.contains_doc(doc_id):
                return seg
        raise KeyError(doc_id)

    def stored(self, doc_id: int) -> dict:
        return self.segment_for(doc_id).stored(doc_id)

    # --------------------------------------------------------------- scoring

    def score_terms(self, terms: Sequence[str], ranker: Ranker) -> np.ndarray:
        """Term-at-a-time scoring into a dense accumulator over global doc IDs.

        Each term's postings are scored as one numpy expression, which is far
        faster in Python than a per-posting loop.
        """
        acc = np.zeros(self.max_doc, dtype=np.float64)
        stats = self.stats
        dls = self.doc_lengths_global
        for term, qtf in Counter(terms).items():
            df = self.doc_freq(term)
            if df == 0:
                continue
            scorer = ranker.term_scorer(df, stats, qtf)
            for p in self.term_postings(term):
                acc[p.doc_ids] += scorer.score_array(p.tfs, dls[p.doc_ids])
        acc[~self.live_mask] = 0.0
        return acc

    @staticmethod
    def top_k(
        acc: np.ndarray, k: int, offset: int = 0, matched: np.ndarray | None = None
    ) -> tuple[np.ndarray, int]:
        """Top ``offset + k`` matched docs by score, in O(n + k log k) via argpartition."""
        if matched is None:
            matched = np.flatnonzero(acc > 0)
        total = len(matched)
        need = min(offset + k, total)
        if need <= 0:
            return np.empty(0, dtype=np.int64), total
        scores = acc[matched]
        if need < total:
            # argpartition alone mishandles ties at the cut-off. Keep everything
            # >= the need-th best score, then sort with a doc-ID tie-break.
            kth = np.partition(-scores, need - 1)[need - 1]
            part = np.flatnonzero(-scores <= kth)
        else:
            part = np.arange(total)
        order = np.lexsort((matched[part], -scores[part]))
        return matched[part][order][offset : offset + k], total

    # ---------------------------------------------------------------- search

    def parse(self, text: str, default_op: str = "OR") -> Node | None:
        return parse_query(text, self.analyzer, default_op)

    def search(
        self,
        text: str,
        k: int = 10,
        ranker: Ranker | None = None,
        offset: int = 0,
        default_op: str = "OR",
        engine: str = "auto",
        use_wand: bool = True,
        load_fields: bool = True,
    ) -> SearchResults:
        """Parse and execute ``text``. See :mod:`minisearch.query.parser` for the syntax."""
        t0 = time.perf_counter()
        ranker = ranker or BM25()
        node = self.parse(text, default_op)
        if node is None or self.max_doc == 0:
            return SearchResults([], 0, (time.perf_counter() - t0) * 1000, [], str(node or ""))

        from minisearch.query.daat_engine import DaatEngine, supports

        stats: dict = {}
        if engine == "daat" and supports(node):
            daat = DaatEngine(self, ranker)
            top, total = daat.execute(node, offset + k, use_wand)
            ranked = top.results()[offset : offset + k]
            stats = {
                "docs_scored": daat.docs_scored,
                "blocks_decoded": daat.stats.blocks_decoded,
                "blocks_skipped": daat.stats.blocks_skipped,
                "postings_visited": daat.stats.postings_visited,
            }
            used = "daat-wand" if use_wand and supports(node) == "or" else "daat"
            exact = total is not None
            total_hits = total if total is not None else daat.docs_scored
        elif engine in ("auto", "vector", "daat"):
            from minisearch.query.vector_engine import VectorEngine

            vec = VectorEngine(self, ranker)
            matched, acc = vec.execute(node)
            ids, total_hits = self.top_k(acc, k, offset, matched)
            ranked = [(int(d), float(acc[d])) for d in ids]
            stats = {"postings_scored": vec.postings_scored}
            used, exact = "vector", True
        else:
            raise ValueError(f"unknown engine {engine!r}")

        hits = [Hit(d, s, self.stored(d) if load_fields else {}) for d, s in ranked]
        return SearchResults(
            hits=hits,
            total_hits=total_hits,
            took_ms=(time.perf_counter() - t0) * 1000,
            query_terms=query_terms(node),
            parsed=str(node),
            engine=used,
            total_is_exact=exact,
            stats=stats,
        )
