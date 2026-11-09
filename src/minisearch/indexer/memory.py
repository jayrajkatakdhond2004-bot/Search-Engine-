"""In-memory segment: the mutable write buffer of the index.

New documents are inverted into a :class:`MemorySegment`. When the buffer
reaches ``max_buffered_docs`` it is *flushed* into an immutable on-disk
segment (see :mod:`minisearch.storage.segment_writer`). Buffered documents
can be searched straight away (near-real-time search), because the buffer
implements the same :class:`Segment` interface as disk segments.
"""

from __future__ import annotations

import bisect

import numpy as np

from minisearch.indexer.segment import Postings, Segment, TermInfo


class _TermBuffer:
    """Growable postings for one term: parallel lists of doc IDs, TFs and positions."""

    __slots__ = ("docs", "tfs", "positions", "max_tf", "min_dl")

    def __init__(self) -> None:
        self.docs: list[int] = []
        self.tfs: list[int] = []
        self.positions: list[list[int]] = []
        self.max_tf = 0
        self.min_dl = 1 << 31


class MemorySegment(Segment):
    """Mutable segment that holds documents until they are flushed to disk."""

    def __init__(self, doc_base: int = 0) -> None:
        self.doc_base = doc_base
        self._postings: dict[str, _TermBuffer] = {}
        self._doc_lengths: list[int] = []
        self._stored: list[dict] = []
        self._deleted: set[int] = set()
        self._sorted_terms: list[str] | None = None
        self._doc_lengths_arr: np.ndarray | None = None
        self.num_tokens = 0

    # ------------------------------------------------------------------ write

    def add(self, analyzed: list[tuple[str, int]], stored: dict) -> int:
        """Invert one analysed document and return its global doc ID."""
        doc_id = self.doc_base + len(self._doc_lengths)
        doc_len = len(analyzed)

        per_term: dict[str, list[int]] = {}
        for term, pos in analyzed:
            lst = per_term.get(term)
            if lst is None:
                per_term[term] = [pos]
            else:
                lst.append(pos)

        postings = self._postings
        for term, positions in per_term.items():
            buf = postings.get(term)
            if buf is None:
                buf = postings[term] = _TermBuffer()
            tf = len(positions)
            buf.docs.append(doc_id)
            buf.tfs.append(tf)
            buf.positions.append(positions)
            if tf > buf.max_tf:
                buf.max_tf = tf
            if doc_len < buf.min_dl:
                buf.min_dl = doc_len

        self._doc_lengths.append(doc_len)
        self._stored.append(stored)
        self.num_tokens += doc_len
        self._sorted_terms = None
        self._doc_lengths_arr = None
        return doc_id

    def delete(self, doc_id: int) -> None:
        if not self.contains_doc(doc_id):
            raise KeyError(doc_id)
        self._deleted.add(doc_id)

    # ------------------------------------------------------------------- read

    @property
    def max_doc(self) -> int:
        return len(self._doc_lengths)

    @property
    def doc_lengths(self) -> np.ndarray:
        if self._doc_lengths_arr is None:
            self._doc_lengths_arr = np.asarray(self._doc_lengths, dtype=np.int32)
        return self._doc_lengths_arr

    @property
    def deleted(self) -> set[int]:
        return self._deleted

    def term_info(self, term: str) -> TermInfo | None:
        buf = self._postings.get(term)
        if buf is None:
            return None
        return TermInfo(df=len(buf.docs), cf=sum(buf.tfs), max_tf=buf.max_tf, min_dl=buf.min_dl)

    def postings(self, term: str) -> Postings | None:
        buf = self._postings.get(term)
        if buf is None:
            return None
        return Postings(
            np.asarray(buf.docs, dtype=np.int64),
            np.asarray(buf.tfs, dtype=np.int32),
            positions=buf.positions,
        )

    def terms(self) -> list[str]:
        if self._sorted_terms is None:
            self._sorted_terms = sorted(self._postings)
        return self._sorted_terms

    def terms_with_prefix(self, prefix: str) -> list[str]:
        terms = self.terms()
        lo = bisect.bisect_left(terms, prefix)
        hi = bisect.bisect_left(terms, prefix + "\U0010ffff")
        return terms[lo:hi]

    def stored(self, doc_id: int) -> dict:
        return self._stored[doc_id - self.doc_base]

    def raw_term_buffers(self) -> dict[str, _TermBuffer]:
        """Direct access for the segment writer (avoids copying into numpy)."""
        return self._postings

    def stored_docs(self) -> list[dict]:
        return self._stored

    def __len__(self) -> int:
        return self.max_doc
