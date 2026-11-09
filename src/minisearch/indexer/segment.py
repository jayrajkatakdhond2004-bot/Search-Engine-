"""The read-side segment abstraction shared by in-memory and on-disk segments.

An index is a list of immutable *segments* plus one mutable in-memory buffer.
Each segment owns a contiguous range of global doc IDs,
``[doc_base, doc_base + max_doc)``. Within a segment, postings are sorted by
doc ID, so concatenating segments in doc_base order gives globally sorted
postings. That makes multi-segment search and segment merging simple.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TermInfo:
    """Dictionary entry statistics for one term in one segment."""

    df: int  # number of documents containing the term
    cf: int  # total occurrences (collection frequency)
    max_tf: int  # max term frequency over the postings (for WAND upper bounds)
    min_dl: int  # min document length over the postings (for WAND upper bounds)


class Postings:
    """Decoded postings of one term within one segment.

    ``doc_ids`` are *global* doc IDs, sorted ascending. Positions are decoded
    lazily, because only phrase queries need them.
    """

    __slots__ = ("doc_ids", "tfs", "_positions", "_flat_loader", "_flat", "_starts")

    def __init__(self, doc_ids: np.ndarray, tfs: np.ndarray, positions=None, flat_loader=None):
        self.doc_ids = doc_ids
        self.tfs = tfs
        self._positions = positions  # list[list[int]] (memory segments)
        self._flat_loader = flat_loader  # () -> flat np.ndarray (disk segments)
        self._flat: np.ndarray | None = None
        self._starts: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.doc_ids)

    def flat_positions(self) -> np.ndarray:
        """All positions concatenated; ``tfs[i]`` of them belong to ``doc_ids[i]``."""
        if self._flat is None:
            if self._flat_loader is not None:
                self._flat = self._flat_loader()
            elif self._positions is not None:
                n = len(self.doc_ids)
                self._flat = np.fromiter(
                    (p for doc in self._positions[:n] for p in doc),
                    dtype=np.int64,
                    count=int(self.tfs.sum()),
                )
            else:
                raise ValueError("positions are not available")
        return self._flat

    def positions(self, i: int) -> np.ndarray:
        """Positions of the term in the ``i``-th document of this postings list."""
        if self._positions is not None:
            return np.asarray(self._positions[i], dtype=np.int64)
        if self._starts is None:
            self._starts = np.concatenate(([0], np.cumsum(self.tfs)))
        flat = self.flat_positions()
        return flat[self._starts[i] : self._starts[i + 1]]


class Segment(ABC):
    """Read interface implemented by :class:`MemorySegment` and :class:`DiskSegment`."""

    doc_base: int

    @property
    @abstractmethod
    def max_doc(self) -> int:
        """Number of doc IDs allocated in this segment (including deleted ones)."""

    @property
    @abstractmethod
    def doc_lengths(self) -> np.ndarray:
        """``doc_lengths[local_id]`` = number of indexed tokens in the doc."""

    @abstractmethod
    def term_info(self, term: str) -> TermInfo | None: ...

    @abstractmethod
    def postings(self, term: str) -> Postings | None: ...

    @abstractmethod
    def terms(self) -> list[str]:
        """All terms, sorted."""

    @abstractmethod
    def stored(self, doc_id: int) -> dict:
        """Stored fields (title, url, body...) for a *global* doc ID."""

    @property
    @abstractmethod
    def deleted(self) -> set[int]:
        """Global doc IDs that are marked deleted (tombstones)."""

    # ----- helpers shared by all segments -----

    @property
    def num_live_docs(self) -> int:
        return self.max_doc - len(self.deleted)

    @property
    def total_length(self) -> int:
        """Sum of live document lengths (for avgdl)."""
        total = int(self.doc_lengths.sum())
        for d in self.deleted:
            total -= int(self.doc_lengths[d - self.doc_base])
        return total

    def contains_doc(self, doc_id: int) -> bool:
        return self.doc_base <= doc_id < self.doc_base + self.max_doc

    def doc_length(self, doc_id: int) -> int:
        return int(self.doc_lengths[doc_id - self.doc_base])

    def terms_with_prefix(self, prefix: str) -> list[str]:
        """Terms starting with ``prefix``. Binary search on the sorted dictionary."""
        import bisect

        terms = self.terms()
        lo = bisect.bisect_left(terms, prefix)
        hi = bisect.bisect_left(terms, prefix + "\U0010ffff")
        return terms[lo:hi]
