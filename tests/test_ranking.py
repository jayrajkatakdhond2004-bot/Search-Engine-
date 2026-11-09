"""Ranking tests. The expected scores below are computed by hand.

Corpus after analysis (Lucene stopwords + Porter):
    d0 "the cat sat on the mat" -> [cat, sat, mat]          |d0| = 3
    d1 "the dog sat"            -> [dog, sat]               |d1| = 2
    d2 "cat cat cat dog"        -> [cat, cat, cat, dog]     |d2| = 4
    N = 3, avgdl = 9 / 3 = 3.0, k1 = 1.2, b = 0.75

IDF(cat) = IDF(dog) = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.4700036292

BM25(t, d) = IDF * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    cat,d0: K = 1.2*(0.25 + 0.75*3/3) = 1.2 -> 0.4700036 * 1*2.2/2.2 = 0.4700036
    cat,d2: K = 1.2*(0.25 + 0.75*4/3) = 1.5 -> 0.4700036 * 3*2.2/4.5 = 0.6893387
    dog,d1: K = 1.2*(0.25 + 0.75*2/3) = 0.9 -> 0.4700036 * 2.2/1.9   = 0.5442147
    dog,d2: K = 1.5                         -> 0.4700036 * 2.2/2.5   = 0.4136032
Query "cat dog": d2 = 1.1029419 > d1 = 0.5442147 > d0 = 0.4700036
"""

import math

import numpy as np
import pytest

from minisearch.analysis import Analyzer
from minisearch.indexer import MemorySegment
from minisearch.query import IndexSearcher
from minisearch.ranking import BM25, CollectionStats, TfIdf, bm25_idf, get_ranker

DOCS = ["the cat sat on the mat", "the dog sat", "cat cat cat dog"]


@pytest.fixture
def searcher():
    analyzer = Analyzer()
    seg = MemorySegment()
    for text in DOCS:
        seg.add(analyzer.analyze(text), {"body": text})
    return IndexSearcher([seg], analyzer)


def test_idf_hand_computed():
    assert bm25_idf(2, 3) == pytest.approx(math.log(1.6))
    assert bm25_idf(2, 3) == pytest.approx(0.4700036292)


def test_bm25_single_term_hand_computed(searcher):
    res = searcher.search("cat", ranker=BM25(k1=1.2, b=0.75))
    assert [h.doc_id for h in res.hits] == [2, 0]
    assert res.hits[0].score == pytest.approx(0.6893386562, rel=1e-9)
    assert res.hits[1].score == pytest.approx(0.4700036292, rel=1e-9)


def test_bm25_multi_term_hand_computed(searcher):
    res = searcher.search("cat dog", ranker=BM25())
    assert [h.doc_id for h in res.hits] == [2, 1, 0]
    assert [h.score for h in res.hits] == pytest.approx([1.1029418, 0.5442147, 0.4700036], abs=1e-6)
    assert res.total_hits == 3


def test_bm25_b_zero_ignores_length(searcher):
    stats = CollectionStats(num_docs=3, total_length=9)
    s = BM25(k1=1.2, b=0.0).term_scorer(2, stats)
    assert s.score(1, 2) == pytest.approx(s.score(1, 400))


def test_bm25_tf_saturates_at_idf_times_k1_plus_1():
    stats = CollectionStats(num_docs=1000, total_length=100_000)
    s = BM25(k1=1.2, b=0.75).term_scorer(10, stats)
    ceiling = bm25_idf(10, 1000) * 2.2
    assert s.score(10_000, 100) < ceiling
    assert s.score(10_000, 100) == pytest.approx(ceiling, rel=1e-3)


def test_vectorised_matches_scalar():
    stats = CollectionStats(num_docs=50, total_length=5000)
    for ranker in (BM25(0.9, 0.4), TfIdf(), TfIdf("log")):
        s = ranker.term_scorer(7, stats, query_tf=2)
        tfs = np.array([1, 2, 5, 9])
        dls = np.array([10, 100, 50, 300])
        expected = [s.score(int(t), int(d)) for t, d in zip(tfs, dls, strict=True)]
        assert s.score_array(tfs, dls) == pytest.approx(expected)


def test_upper_bound_dominates_all_postings():
    rng = np.random.default_rng(0)
    stats = CollectionStats(num_docs=100, total_length=10_000)
    for ranker in (BM25(), TfIdf()):
        s = ranker.term_scorer(5, stats)
        tfs = rng.integers(1, 20, 500)
        dls = rng.integers(5, 400, 500)
        ub = s.upper_bound(int(tfs.max()), int(dls.min()))
        assert (s.score_array(tfs, dls) <= ub + 1e-12).all()


def test_tfidf_hand_computed(searcher):
    # idf = 1 + ln(4/3); score = sqrt(tf) * idf^2 / sqrt(|d|)
    idf = 1 + math.log(4 / 3)
    res = searcher.search("cat", ranker=TfIdf())
    scores = {h.doc_id: h.score for h in res.hits}
    assert scores[0] == pytest.approx(1 * idf**2 / math.sqrt(3))
    assert scores[2] == pytest.approx(math.sqrt(3) * idf**2 / math.sqrt(4))


def test_rankers_are_pluggable(searcher):
    for name in ("bm25", "tfidf"):
        assert searcher.search("cat", ranker=get_ranker(name)).hits
    with pytest.raises(ValueError):
        get_ranker("pagerank")


def test_global_stats_across_segments():
    """Two segments must score exactly like one segment holding the same docs."""
    analyzer = Analyzer()
    one = MemorySegment()
    a, b = MemorySegment(0), MemorySegment(2)
    for i, text in enumerate(DOCS):
        one.add(analyzer.analyze(text), {})
        (a if i < 2 else b).add(analyzer.analyze(text), {})
    r1 = IndexSearcher([one], analyzer).search("cat dog")
    r2 = IndexSearcher([a, b], analyzer).search("cat dog")
    assert [(h.doc_id, round(h.score, 12)) for h in r1.hits] == [
        (h.doc_id, round(h.score, 12)) for h in r2.hits
    ]


def test_deleted_docs_are_not_returned(searcher):
    searcher.segments[0].delete(2)
    searcher._live_mask = None
    searcher._stats = None
    assert [h.doc_id for h in searcher.search("cat").hits] == [0]


def test_pagination_offsets(searcher):
    page1 = searcher.search("cat dog", k=2)
    page2 = searcher.search("cat dog", k=2, offset=2)
    assert [h.doc_id for h in page1.hits] == [2, 1]
    assert [h.doc_id for h in page2.hits] == [0]
