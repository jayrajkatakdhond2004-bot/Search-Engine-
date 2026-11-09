"""Postings iterators with skip pointers (document-at-a-time access).

A :class:`TermIterator` walks one term's postings across all segments in
doc ID order, and supports two operations:

* ``next()``: move to the next posting.
* ``advance(target)``: move to the first posting with ``doc >= target``.

``advance`` is where the skip pointers pay off. Each disk segment stores a
skip table with the last doc ID of every 128-posting block. To advance far
ahead, the iterator binary-searches the *skip table*, not the postings,
and decodes only the one block that can contain ``target``. Every block
jumped over is never decompressed. Intersecting a rare term (10 postings)
with a common one (1M postings) then decodes at most 10 blocks of the
common list, instead of all ~7,800.

Blocks are converted to Python lists when loaded: indexing a list from Python
is ~5x faster than indexing a numpy array element by element.
"""

from __future__ import annotations

import bisect
import sys
from dataclasses import dataclass

from minisearch.indexer.segment import Segment
from minisearch.storage.disk_segment import DiskSegment

END = sys.maxsize


@dataclass
class IterStats:
    blocks_decoded: int = 0
    blocks_skipped: int = 0
    postings_visited: int = 0


class _Block:
    """A lazily decodable block: knows its last doc before it is decoded."""

    __slots__ = ("last_doc", "_loader", "docs", "tfs")

    def __init__(self, last_doc: int, loader) -> None:
        self.last_doc = last_doc
        self._loader = loader
        self.docs: list[int] | None = None
        self.tfs: list[int] | None = None

    def load(self) -> None:
        if self.docs is None:
            docs, tfs = self._loader()
            self.docs, self.tfs = docs.tolist(), tfs.tolist()


def _segment_blocks(seg: Segment, term: str) -> list[_Block]:
    if isinstance(seg, DiskSegment):
        entry = seg.entry(term)
        if entry is None:
            return []
        files, base = seg.files, seg.doc_base
        skip = files.skip_table(entry)
        if skip is None:

            def load_all(e=entry):
                d, t = files.decode_postings(e)
                return d + base, t

            # Last doc is unknown without decoding. A single block is decoded on first touch anyway.
            blk = _Block(END - 1, load_all)
            blk.load()
            blk.last_doc = blk.docs[-1]
            return [blk]
        blocks = []
        for b, last in enumerate(skip[:, 0].tolist()):

            def load(b=b, e=entry):
                d, t = files.decode_block(e, b)
                return d + base, t

            blocks.append(_Block(last + base, load))
        return blocks
    p = seg.postings(term)
    if p is None or len(p) == 0:
        return []
    blk = _Block(int(p.doc_ids[-1]), lambda p=p: (p.doc_ids, p.tfs))
    return [blk]


class TermIterator:
    """Iterates one term's postings across segments, using skip tables to advance."""

    __slots__ = ("term", "blocks", "last_docs", "b", "i", "doc", "_docs", "_tfs", "stats", "cost")

    def __init__(self, segments: list[Segment], term: str, stats: IterStats | None = None) -> None:
        self.term = term
        self.blocks: list[_Block] = []
        for seg in segments:  # segments are in doc-base order
            self.blocks.extend(_segment_blocks(seg, term))
        self.last_docs = [b.last_doc for b in self.blocks]
        self.stats = stats or IterStats()
        self.cost = sum(s.term_info(term).df for s in segments if s.term_info(term) is not None)
        self.b = -1
        self.i = 0
        self._docs: list[int] = []
        self._tfs: list[int] = []
        self.doc = -1
        self._enter_block(0)

    def _enter_block(self, b: int) -> None:
        if b >= len(self.blocks):
            self.b = len(self.blocks)
            self.doc = END
            return
        blk = self.blocks[b]
        if blk.docs is None:
            blk.load()
            self.stats.blocks_decoded += 1
        self.b, self.i = b, 0
        self._docs, self._tfs = blk.docs, blk.tfs
        self.doc = self._docs[0]
        self.stats.postings_visited += 1

    def next(self) -> int:
        self.i += 1
        if self.i < len(self._docs):
            self.doc = self._docs[self.i]
            self.stats.postings_visited += 1
        else:
            self._enter_block(self.b + 1)
        return self.doc

    def advance(self, target: int) -> int:
        """Move to the first doc ``>= target`` (no-op if already there)."""
        if self.doc >= target:
            return self.doc
        if target > self.last_docs[self.b]:
            # Skip pointers: binary search block boundaries, decode only the target block.
            nb = bisect.bisect_left(self.last_docs, target, self.b + 1)
            self.stats.blocks_skipped += max(0, nb - self.b - 1)
            self._enter_block(nb)
            if self.doc >= target or self.doc == END:
                return self.doc
        self.i = bisect.bisect_left(self._docs, target, self.i + 1)
        self.doc = self._docs[self.i]  # guaranteed: target <= block's last doc
        self.stats.postings_visited += 1
        return self.doc

    def tf(self) -> int:
        return self._tfs[self.i]
