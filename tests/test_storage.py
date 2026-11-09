"""On-disk segments, persistence, deletes and merging."""

import numpy as np
import pytest

from conftest import make_corpus
from minisearch import Index
from minisearch.indexer.memory import MemorySegment
from minisearch.indexer.merge import LogMergePolicy
from minisearch.storage.disk_segment import SegmentFiles
from minisearch.storage.format import BLOCK_SIZE, CorruptIndexError, SegmentMeta


def ranked(index: Index, q: str, k: int = 20):
    return [(h.fields["id"], round(h.score, 9)) for h in index.searcher().search(q, k=k).hits]


QUERIES = ["search engine", "posting segment merge", "checksum", "byte block skip heap", "zzz"]


@pytest.mark.parametrize("codec", ["vbyte", "raw"])
def test_disk_segment_matches_memory_segment(tmp_path, corpus, codec):
    idx = Index(tmp_path / "i", codec=codec, max_buffered_docs=10**9)
    idx.add_many(corpus)
    mem: MemorySegment = idx._buffer
    expected = {t: mem.postings(t) for t in mem.terms()}
    expected_lengths = mem.doc_lengths.copy()
    idx.commit()
    (disk,) = idx.searcher().segments

    assert disk.terms() == sorted(expected)
    assert np.array_equal(disk.doc_lengths, expected_lengths)
    saw_blocked = False
    for term, p in expected.items():
        d = disk.postings(term)
        assert np.array_equal(d.doc_ids, p.doc_ids), term
        assert np.array_equal(d.tfs, p.tfs), term
        for i in range(len(p)):
            assert list(d.positions(i)) == list(p.positions(i)), term
        info, minfo = disk.term_info(term), mem.term_info(term)
        assert info == minfo
        saw_blocked |= len(p) > BLOCK_SIZE
    assert saw_blocked, "corpus should exercise multi-block postings with skip tables"
    assert disk.term_info("not-a-term") is None and disk.postings("aaaa") is None


def test_block_decoding_with_skip_table(tmp_path, corpus):
    idx = Index(tmp_path / "i")
    idx.add_many(corpus)
    idx.commit()
    (seg,) = idx.searcher().segments
    term = max(seg.terms(), key=lambda t: seg.term_info(t).df)
    entry = seg.entry(term)
    docs, tfs = seg.files.decode_postings(entry)
    skip = seg.files.skip_table(entry)
    assert skip is not None and len(skip) == -(-len(docs) // BLOCK_SIZE)
    for b in range(len(skip)):
        bd, bt = seg.files.decode_block(entry, b)
        assert np.array_equal(bd, docs[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE])
        assert np.array_equal(bt, tfs[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE])
        assert int(skip[b, 0]) == int(bd[-1])  # skip entry = last doc in block


def test_search_results_identical_before_and_after_flush(tmp_path, corpus):
    idx = Index(tmp_path / "i")
    idx.add_many(corpus)
    before = {q: ranked(idx, q) for q in QUERIES}
    idx.commit()
    assert {q: ranked(idx, q) for q in QUERIES} == before


def test_persistence_and_reopen(tmp_path, corpus):
    path = tmp_path / "i"
    with Index(path) as idx:
        idx.add_many(corpus)
        expected = {q: ranked(idx, q) for q in QUERIES}
    reopened = Index(path)
    assert len(reopened) == len(corpus)
    assert {q: ranked(reopened, q) for q in QUERIES} == expected
    assert reopened.searcher().stored(0)["id"] == "d0"


def test_prefix_terms_from_front_coded_dictionary(tmp_path, corpus):
    idx = Index(tmp_path / "i")
    idx.add_many(corpus)
    idx.commit()
    (seg,) = idx.searcher().segments
    all_terms = seg.terms()
    for prefix in ["s", "se", "comp", "b", "zz", ""]:
        assert seg.terms_with_prefix(prefix) == [t for t in all_terms if t.startswith(prefix)]


def test_incremental_indexing_creates_segments_and_merges(tmp_path):
    docs = make_corpus(1200, seed=1)
    policy = LogMergePolicy(merge_factor=3, min_segment_docs=100)
    idx = Index(tmp_path / "i", max_buffered_docs=100, merge_policy=policy)
    idx.add_many(docs)
    idx.commit()
    # 12 flushes of 100 docs, merge factor 3 -> segments are merged into larger tiers.
    sizes = [m.max_doc for m in idx.manifest.segments]
    assert sum(sizes) == 1200
    assert len(sizes) < 12
    assert max(sizes) >= 300

    reference = Index(tmp_path / "ref", max_buffered_docs=10**9)
    reference.add_many(docs)
    reference.commit()
    for q in QUERIES:
        assert ranked(idx, q) == ranked(reference, q)


def test_force_merge_gives_identical_ranking(tmp_path):
    docs = make_corpus(500, seed=2)
    idx = Index(tmp_path / "i", max_buffered_docs=50, auto_merge=False)
    idx.add_many(docs)
    idx.commit()
    assert len(idx.manifest.segments) == 10
    before = {q: ranked(idx, q) for q in QUERIES}
    idx.force_merge()
    assert len(idx.manifest.segments) == 1
    assert {q: ranked(idx, q) for q in QUERIES} == before


def test_deletes_updates_and_merge_expunge(tmp_path):
    docs = make_corpus(300, seed=3)
    path = tmp_path / "i"
    idx = Index(path, max_buffered_docs=100, auto_merge=False)
    idx.add_many(docs)
    idx.commit()
    for i in range(0, 300, 3):
        assert idx.delete(f"d{i}")
    assert not idx.delete("missing")
    idx.add({"id": "d1", "title": "replacement", "body": "brand new unique wording"})
    idx.commit()

    survivors = [d for i, d in enumerate(docs) if i % 3 and i != 1]
    survivors.append({"id": "d1", "title": "replacement", "body": "brand new unique wording"})
    hits = idx.searcher().search("search engine", k=1000).hits
    assert all(int(h.fields["id"][1:]) % 3 for h in hits)

    # Deletes persist across reopen.
    idx = Index(path, auto_merge=False)
    assert len(idx) == 200
    assert idx.searcher().search("unique wording").hits[0].fields["id"] == "d1"

    # After expunging deletes, scoring equals a fresh index of the survivors (df exact again).
    idx.force_merge()
    assert idx.stats()["deleted_documents"] == 0
    fresh = Index(tmp_path / "fresh")
    fresh.add_many(survivors)
    fresh.commit()
    for q in QUERIES + ["unique wording"]:
        assert sorted(ranked(idx, q, k=500)) == sorted(ranked(fresh, q, k=500))


def test_snapshot_isolation(tmp_path, corpus):
    idx = Index(tmp_path / "i", max_buffered_docs=200)
    idx.add_many(corpus[:300])
    snap = idx.searcher()
    n_before = snap.stats.num_docs
    res_before = snap.search("search engine", k=50)
    idx.add_many(corpus[300:])
    idx.delete("d0")
    idx.commit()
    assert snap.stats.num_docs == n_before
    assert [h.doc_id for h in snap.search("search engine", k=50).hits] == [
        h.doc_id for h in res_before.hits
    ]
    assert idx.searcher().stats.num_docs == len(corpus) - 1


def test_compression_reduces_size(tmp_path, corpus):
    sizes = {}
    for codec in ("raw", "vbyte"):
        idx = Index(tmp_path / codec, codec=codec)
        idx.add_many(corpus)
        idx.commit()
        sizes[codec] = idx.stats()["postings_bytes"]
    assert sizes["vbyte"] * 2.5 < sizes["raw"]
    assert idx.stats()["compression_ratio"] > 2.5


def test_corruption_is_detected(tmp_path, corpus):
    idx = Index(tmp_path / "i")
    idx.add_many(corpus[:50])
    idx.commit()
    meta: SegmentMeta = idx.manifest.segments[0]
    post = tmp_path / "i" / f"{meta.name}.post"
    data = bytearray(post.read_bytes())
    data[40] ^= 0xFF
    post.write_bytes(bytes(data))
    with pytest.raises(CorruptIndexError):
        SegmentFiles(tmp_path / "i", meta, verify=True)
    data[0:4] = b"XXXX"
    post.write_bytes(bytes(data))
    with pytest.raises(CorruptIndexError):
        SegmentFiles(tmp_path / "i", meta)


def test_orphan_files_are_garbage_collected(tmp_path):
    path = tmp_path / "i"
    Index(path).close()
    (path / "seg_999999.post").write_bytes(b"junk")
    Index(path)
    assert not (path / "seg_999999.post").exists()


def test_analyzer_mismatch_rejected(tmp_path):
    from minisearch import AnalyzerConfig

    Index(tmp_path / "i", analyzer_config=AnalyzerConfig(stemmer=None)).close()
    with pytest.raises(ValueError):
        Index(tmp_path / "i", analyzer_config=AnalyzerConfig(stemmer="porter"))
    assert Index(tmp_path / "i").analyzer.config.stemmer is None


def test_phrase_cannot_cross_title_body_boundary(tmp_path):
    idx = Index(tmp_path / "i")
    idx.add({"id": "x", "title": "red", "body": "car"})
    p_red = idx.searcher().segments[0].postings("red")
    p_car = idx.searcher().segments[0].postings("car")
    assert p_car.positions(0)[0] - p_red.positions(0)[0] > 1
