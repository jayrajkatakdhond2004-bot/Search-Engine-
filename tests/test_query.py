"""Query parser, boolean / phrase / wildcard semantics, and engine equivalence."""

import random
import re

import numpy as np
import pytest

from conftest import WORDS, make_corpus
from minisearch import BM25, Index, TfIdf
from minisearch.analysis import Analyzer
from minisearch.query import QuerySyntaxError, parse, parse_query

A = Analyzer()


def p(q: str, op: str = "OR") -> str:
    return str(parse_query(q, A, op))


# ------------------------------------------------------------------ parser


@pytest.mark.parametrize(
    "query,expected",
    [
        ("cats", "cat"),
        ("cats dogs", "(cat OR dog)"),
        ("cats AND dogs", "(cat AND dog)"),
        ("cats && dogs || birds", "((cat AND dog) OR bird)"),
        ("x1 y1 AND z1", "(x1 OR (y1 AND z1))"),
        ("NOT cats", "(*:* AND NOT cat)"),
        ("-cats dogs", "(NOT cat OR dog)"),
        ("dogs -cats", "(dog OR NOT cat)"),
        ("dogs AND -cats", "(dog AND NOT cat)"),
        ("(cats OR dogs) AND NOT birds", "((cat OR dog) AND NOT bird)"),
        ('"New York" city', '("new york" OR citi)'),
        ('"university of london"', '"univers london"'),
        ("comput*", "comput*"),
        ("Te?t", "te?t"),
        ("e-mail", '"e mail"'),
        ("NOT NOT cats", "cat"),
        ("the AND cats", "cat"),
        ("", "None"),
        ("the of and", "None"),
    ],
)
def test_parse_and_normalise(query, expected):
    assert p(query) == expected


def test_precedence_or_lt_implicit_lt_and():
    assert p("a1 b1 OR c1 AND d1") == "(a1 OR b1 OR (c1 AND d1))"  # nested ORs flatten
    assert p("a1 b1 OR c1 AND d1", "AND") == "((a1 AND b1) OR (c1 AND d1))"
    assert p("x1 AND y1 z1", "AND") == "(x1 AND y1 AND z1)"


def test_phrase_offsets_keep_stopword_gaps():
    node = parse_query('"bank of the river"', A)
    assert node.terms == ("bank", "river") and node.offsets == (0, 3)


def test_lowercase_operators_are_words():
    assert p("cats and dogs") == "(cat OR dog)"  # 'and' is a stopword


@pytest.mark.parametrize("bad", ["cats AND", "OR cats", "(cats", ")cats", "cats )", "AND"])
def test_syntax_errors(bad):
    if bad == "(cats":  # a missing ')' at the end is tolerated
        assert p(bad) == "cat"
        return
    with pytest.raises(QuerySyntaxError):
        parse(bad)


def test_unterminated_quote_is_lenient():
    assert p('"search engine') == '"search engin"'


# ---------------------------------------------------------- brute-force oracle


class Oracle:
    """Answers queries by scanning every document directly. Slow, obviously correct."""

    def __init__(self, idx: Index, docs: list[dict]):
        self.docs = {}
        for d in docs:
            if d["id"] in idx:
                analyzed = idx._analyze_doc(d["title"], d["body"])
                pos: dict[str, set[int]] = {}
                for t, i in analyzed:
                    pos.setdefault(t, set()).add(i)
                self.docs[d["id"]] = pos

    def term(self, t):
        return {d for d, pos in self.docs.items() if t in pos}

    def phrase(self, terms, offsets):
        out = set()
        for d, pos in self.docs.items():
            if all(t in pos for t in terms):
                starts = {p - offsets[0] for p in pos[terms[0]]}
                for t, o in zip(terms[1:], offsets[1:], strict=True):
                    starts &= {p - o for p in pos[t]}
                if starts:
                    out.add(d)
        return out

    def wildcard(self, pattern):
        rx = re.compile(pattern.replace("*", ".*").replace("?", "."))
        return {d for d, pos in self.docs.items() if any(rx.fullmatch(t) for t in pos)}

    def all(self):
        return set(self.docs)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    docs = make_corpus(800, seed=11, max_len=40)
    idx = Index(tmp_path_factory.mktemp("q") / "i", max_buffered_docs=150, auto_merge=False)
    idx.add_many(docs)
    for i in range(0, 800, 7):
        idx.delete(f"d{i}")
    idx.commit()
    idx.add_many(make_corpus(60, seed=12))  # unflushed buffer docs, ids d0..d59 overwrite
    all_docs = [d for d in docs if d["id"] not in {f"d{i}" for i in range(60)}]
    all_docs += make_corpus(60, seed=12)
    return idx, Oracle(idx, all_docs)


def ids(idx, q, **kw):
    res = idx.searcher().search(q, k=10_000, **kw)
    return {h.fields["id"] for h in res.hits}


def test_index_has_multiple_segments_deletes_and_buffer(built):
    idx, _ = built
    segs = idx.searcher().segments
    assert len(segs) >= 6 and any(s.deleted for s in segs) and len(idx._buffer) == 60


@pytest.mark.parametrize(
    "query,oracle",
    [
        ("search", lambda o: o.term("search")),
        ("search AND engine", lambda o: o.term("search") & o.term("engin")),
        ("search OR checksum", lambda o: o.term("search") | o.term("checksum")),
        ("search AND NOT engine", lambda o: o.term("search") - o.term("engin")),
        ("NOT search", lambda o: o.all() - o.term("search")),
        (
            "(checksum OR footer) AND NOT (search OR engine)",
            lambda o: (o.term("checksum") | o.term("footer")) - o.term("search") - o.term("engin"),
        ),
        ("checksum OR NOT search", lambda o: o.term("checksum") | (o.all() - o.term("search"))),
        ('"search engine"', lambda o: o.phrase(("search", "engin"), (0, 1))),
        ('"engine search index"', lambda o: o.phrase(("engin", "search", "index"), (0, 1, 2))),
        ('"search search"', lambda o: o.phrase(("search", "search"), (0, 1))),
        (
            '"search engine" AND NOT query',
            lambda o: o.phrase(("search", "engin"), (0, 1)) - o.term("queri"),
        ),
        ("co*", lambda o: o.wildcard("co*")),
        ("*ss", lambda o: o.wildcard("*ss")),
        ("c?ecksum", lambda o: o.wildcard("c?ecksum")),
        ("p*t*", lambda o: o.wildcard("p*t*")),
        ("zzzz OR qqqq", lambda o: set()),
    ],
)
def test_boolean_phrase_wildcard_match_oracle(built, query, oracle):
    idx, o = built
    assert ids(idx, query) == oracle(o)


def test_random_boolean_queries_match_oracle(built):
    idx, o = built
    rng = random.Random(5)
    stems = {w: A.terms(w)[0] for w in WORDS}

    def gen(depth):
        if depth == 0 or rng.random() < 0.3:
            w = rng.choice(WORDS)
            return w, o.term(stems[w])
        op = rng.choice(["AND", "OR", "AND NOT"])
        (lq, ls), (rq, rs) = gen(depth - 1), gen(depth - 1)
        if op == "AND":
            return f"({lq} AND {rq})", ls & rs
        if op == "OR":
            return f"({lq} OR {rq})", ls | rs
        return f"({lq} AND NOT {rq})", ls - rs

    for _ in range(60):
        q, expected = gen(3)
        assert ids(idx, q) == expected, q


# ---------------------------------------------------------- engine equivalence


def ranking(idx, q, engine, ranker, k=10, **kw):
    res = idx.searcher().search(q, k=k, engine=engine, ranker=ranker, **kw)
    return [(h.doc_id, round(h.score, 9)) for h in res.hits], res


@pytest.mark.parametrize("ranker", [BM25(), BM25(k1=2.0, b=0.3), TfIdf()])
def test_wand_equals_exhaustive_on_random_or_queries(built, ranker):
    idx, _ = built
    rng = random.Random(1)
    for _ in range(40):
        q = " ".join(rng.sample(WORDS, rng.randint(1, 6)))
        k = rng.choice([1, 5, 10, 50])
        vec, _ = ranking(idx, q, "vector", ranker, k)
        wand, res = ranking(idx, q, "daat", ranker, k)
        exhaustive, _ = ranking(idx, q, "daat", ranker, k, use_wand=False)
        assert res.engine == "daat-wand"
        assert wand == vec == exhaustive, q


def test_wand_skips_work(built):
    idx, _ = built
    q = " ".join(WORDS[:8])  # frequent terms
    _, wand = ranking(idx, q, "daat", BM25(), 10)
    _, full = ranking(idx, q, "daat", BM25(), 10, use_wand=False)
    assert wand.stats["docs_scored"] < full.stats["docs_scored"]
    assert not wand.total_is_exact and full.total_is_exact


def test_conjunction_engine_equals_vector(built):
    idx, _ = built
    rng = random.Random(2)
    for _ in range(40):
        pos = rng.sample(WORDS, rng.randint(1, 3))
        neg = rng.sample([w for w in WORDS if w not in pos], rng.randint(0, 2))
        q = " AND ".join(pos + [f"NOT {w}" for w in neg])
        vec, vres = ranking(idx, q, "vector", BM25(), 20)
        daat, dres = ranking(idx, q, "daat", BM25(), 20)
        assert daat == vec, q
        assert dres.total_hits == vres.total_hits


def test_skip_pointers_skip_blocks(tmp_path):
    # A rare term ANDed with a very common one: the common list must be skipped through.
    docs = [
        {"id": str(i), "body": "common " + ("rare" if i % 997 == 0 else "")} for i in range(5000)
    ]
    idx = Index(tmp_path / "i")
    idx.add_many(docs)
    idx.commit()
    res = idx.searcher().search("rare AND common", engine="daat", k=100)
    assert res.total_hits == 6
    common_blocks = -(-5000 // 128)
    assert res.stats["blocks_decoded"] <= 6 + 1 + 1  # <= one block per match (+rare, +first)
    assert res.stats["blocks_skipped"] > common_blocks // 2


def test_pagination_is_consistent(built):
    idx, _ = built
    full = idx.searcher().search("search engine query", k=30).hits
    pages = []
    for page in range(3):
        pages += idx.searcher().search("search engine query", k=10, offset=10 * page).hits
    assert [h.doc_id for h in pages] == [h.doc_id for h in full]


def test_ties_are_broken_by_doc_id(tmp_path):
    idx = Index(tmp_path / "i")
    idx.add_many([{"id": str(i), "body": "same text"} for i in range(50)])
    for engine in ("vector", "daat"):
        hits = idx.searcher().search("same", k=7, engine=engine).hits
        assert [h.doc_id for h in hits] == list(range(7))


def test_phrase_scores_use_phrase_frequency(tmp_path):
    idx = Index(tmp_path / "i")
    idx.add({"id": "once", "body": "red car blue bike"})
    idx.add({"id": "twice", "body": "red car red car"})
    idx.add({"id": "scattered", "body": "car red bike red car"})
    res = idx.searcher().search('"red car"')
    assert [h.fields["id"] for h in res.hits] == ["twice", "once", "scattered"]


def test_wildcard_is_dismax_scored(tmp_path):
    from minisearch import AnalyzerConfig

    idx = Index(tmp_path / "i", analyzer_config=AnalyzerConfig(stemmer=None))
    idx.add({"id": "variants", "body": "compute computer computing computation"})
    idx.add({"id": "single", "body": "computer computer"})
    idx.add({"id": "other", "body": "nothing relevant here"})
    s = idx.searcher()
    res = {h.fields["id"]: h.score for h in s.search("comput*").hits}
    assert set(res) == {"variants", "single"}
    # Dis-max: a doc scores its best-matching expansion, not the sum over variants.
    for doc in ("variants", "single"):
        per_term = [
            h.score
            for t in ("compute", "computer", "computing", "computation")
            for h in s.search(t).hits
            if h.fields["id"] == doc
        ]
        assert res[doc] == pytest.approx(max(per_term))


def test_query_terms_for_highlighting(built):
    idx, _ = built
    res = idx.searcher().search('"search engine" OR query -checksum')
    assert res.query_terms == ["queri", "search", "engin"]


def test_empty_index_and_query(tmp_path):
    idx = Index(tmp_path / "i")
    assert idx.searcher().search("anything").hits == []
    idx.add({"id": "1", "body": "hello"})
    assert idx.searcher().search("").hits == []
    assert np.all(idx.searcher().live_mask)
