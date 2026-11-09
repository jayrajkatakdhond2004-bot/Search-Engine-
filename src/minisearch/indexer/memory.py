"""In-memory segment: the mutable write buffer of the index.

New documents are inverted into a :class:`MemorySegment`. When the buffer
reaches ``max_buffered_docs`` it is *flushed* to an immutable on-disk segment.
Buffered documents can be searched immediately (near-real-time search) via
:meth:`MemorySegment.snapshot`: a frozen view of the first ``n`` documents.

Postings lists only grow by appending, and a doc's postings are complete
before ``add`` returns. A view that reads only doc IDs ``< n`` is therefore
safe while more documents are being added.
"""

from __future__ import annotations

import bisect
from itertools import chain

import numpy as np

from minisearch.indexer.segment import Postings, Segment, TermInfo


class _TermBuffer:
    """Growable postings for one term: parallel lists of local doc IDs, TFs and positions."""

    __slots__ = ("docs", "tfs", "positions", "max_tf", "min_dl")

    def __init__(self) -> None:
        self.docs: list[int] = []
        self.tfs: list[int] = []
        self.positions: list[list[int]] = []
        self.max_tf = 0
        self.min_dl = 1 << 31


class _MemoryReader(Segment):
    """Read logic shared by the live segment and its snapshots."""

    _seg: MemorySegment
    _upto: int | None  # None = live (all docs)
    _deleted_local: set[int] | frozenset[int]

    @property
    def max_doc(self) -> int:
        return len(self._seg._doc_lengths) if self._upto is None else self._upto

    @property
    def doc_lengths(self) -> np.ndarray:
        return self._seg._lengths_array()[: self.max_doc]

    @property
    def deleted(self) -> set[int]:
        base = self.doc_base
        return {base + d for d in self._deleted_local}

    def _cut(self, buf: _TermBuffer) -> int:
        if self._upto is None:
            return len(buf.docs)
        return bisect.bisect_left(buf.docs, self._upto)

    def term_info(self, term: str) -> TermInfo | None:
        buf = self._seg._postings.get(term)
        if buf is None:
            return None
        n = self._cut(buf)
        if n == 0:
            return None
        if n == len(buf.docs):
            return TermInfo(len(buf.docs), sum(buf.tfs), buf.max_tf, buf.min_dl)
        tfs = buf.tfs[:n]
        lengths = self._seg._doc_lengths
        return TermInfo(n, sum(tfs), max(tfs), min(lengths[d] for d in buf.docs[:n]))

    def postings(self, term: str) -> Postings | None:
        buf = self._seg._postings.get(term)
        if buf is None:
            return None
        n = self._cut(buf)
        if n == 0:
            return None
        docs = np.asarray(buf.docs[:n], dtype=np.int64) + self.doc_base
        return Postings(docs, np.asarray(buf.tfs[:n], dtype=np.int64), positions=buf.positions)

    def terms(self) -> list[str]:
        if self._upto is None or self._upto == len(self._seg._doc_lengths):
            return self._seg._sorted()
        return [t for t in self._seg._sorted() if self._cut(self._seg._postings[t]) > 0]

    def terms_with_prefix(self, prefix: str) -> list[str]:
        terms = self._seg._sorted()
        lo = bisect.bisect_left(terms, prefix)
        hi = bisect.bisect_left(terms, prefix + "\U0010ffff")
        return [t for t in terms[lo:hi] if self._cut(self._seg._postings[t]) > 0]

    def stored(self, doc_id: int) -> dict:
        return self._seg._stored[doc_id - self.doc_base]


class MemorySegment(_MemoryReader):
    """Mutable segment that buffers documents until they are flushed to disk."""

    def __init__(self, doc_base: int = 0) -> None:
        self.doc_base = doc_base
        self._seg = self
        self._upto = None
        self._postings: dict[str, _TermBuffer] = {}
        self._doc_lengths: list[int] = []
        self._stored: list[dict] = []
        self._ids: list[str] = []
        self._deleted_local: set[int] = set()
        self._sorted_terms: list[str] | None = None
        self._lengths_arr: np.ndarray | None = None
        self.num_tokens = 0

    # ------------------------------------------------------------------ write

    def add(self, analyzed: list[tuple[str, int]], stored: dict, ext_id: str = "") -> int:
        """Invert one analysed document. Returns its *global* doc ID."""
        local = len(self._doc_lengths)
        doc_len = len(analyzed)

        per_term: dict[str, list[int]] = {}
        for term, pos in analyzed:
            lst = per_term.get(term)
            if lst is None:
                per_term[term] = [pos]
            else:
                lst.append(pos)

        postings = self._postings
        new_terms = False
        for term, positions in per_term.items():
            buf = postings.get(term)
            if buf is None:
                buf = postings[term] = _TermBuffer()
                new_terms = True
            tf = len(positions)
            buf.tfs.append(tf)
            buf.positions.append(positions)
            if tf > buf.max_tf:
                buf.max_tf = tf
            if doc_len < buf.min_dl:
                buf.min_dl = doc_len
            buf.docs.append(local)  # appended last: readers bound by doc ID see a complete posting

        self._stored.append(stored)
        self._ids.append(ext_id)
        self._doc_lengths.append(doc_len)
        self.num_tokens += doc_len
        if new_terms:
            self._sorted_terms = None
        self._lengths_arr = None
        return self.doc_base + local

    def delete_local(self, local: int) -> None:
        if not 0 <= local < len(self._doc_lengths):
            raise KeyError(local)
        self._deleted_local.add(local)

    def delete(self, doc_id: int) -> None:
        self.delete_local(doc_id - self.doc_base)

    # ------------------------------------------------------------------ views

    def snapshot(self, doc_base: int | None = None) -> _MemorySnapshot:
        """Frozen view of the documents buffered so far."""
        return _MemorySnapshot(
            self,
            self.doc_base if doc_base is None else doc_base,
            len(self._doc_lengths),
            frozenset(self._deleted_local),
        )

    # --------------------------------------------------------------- internal

    def _sorted(self) -> list[str]:
        if self._sorted_terms is None:
            self._sorted_terms = sorted(self._postings)
        return self._sorted_terms

    def _lengths_array(self) -> np.ndarray:
        if self._lengths_arr is None or len(self._lengths_arr) != len(self._doc_lengths):
            self._lengths_arr = np.asarray(self._doc_lengths, dtype=np.int64)
        return self._lengths_arr

    # ---------------------------------------------------------- flush support

    def iter_for_flush(self):
        """Yield ``(term, docs, tfs, flat_positions, min_dl)`` in sorted term order."""
        for term in self._sorted():
            buf = self._postings[term]
            yield term, buf.docs, buf.tfs, list(chain.from_iterable(buf.positions)), buf.min_dl

    @property
    def ids(self) -> list[str]:
        return self._ids

    @property
    def stored_docs(self) -> list[dict]:
        return self._stored

    @property
    def deleted_local(self) -> set[int]:
        return self._deleted_local

    def __len__(self) -> int:
        return len(self._doc_lengths)


class _MemorySnapshot(_MemoryReader):
    def __init__(self, seg: MemorySegment, doc_base: int, upto: int, deleted: frozenset[int]):
        self._seg = seg
        self.doc_base = doc_base
        self._upto = upto
        self._deleted_local = deleted
