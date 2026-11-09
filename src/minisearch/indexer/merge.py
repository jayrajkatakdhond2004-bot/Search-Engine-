"""Segment flushing, merging, and the merge policy.

Why segments?
-------------
An inverted index cannot be updated in place cheaply: inserting a doc ID in
the middle of a compressed postings list means rewriting it. Like Lucene,
minisearch writes *immutable* segments and merges them in the background.
Each merge is a sequential k-way merge with no random I/O.

Merge policy (logarithmic tiers)
--------------------------------
A segment's *tier* is ``floor(log_F(live_docs / min_size))``, with merge
factor ``F`` (default 10). When ``F`` adjacent segments share a tier, they
are merged into one segment of the next tier. Each document is rewritten
about ``log_F(N / min_size)`` times over the life of the index, so total
merge cost is O(N log N), while the segment count (and with it per-query
overhead) stays O(F log N). Segments where more than ``max_deleted_ratio``
of the docs are deleted are also rewritten, to reclaim space and make df
exact again.

Only *adjacent* segments are merged. Doc IDs then stay in insertion order
across the whole index, which keeps postings concatenation trivial.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minisearch.indexer.memory import MemorySegment
from minisearch.storage.disk_segment import DiskSegment
from minisearch.storage.format import SegmentMeta
from minisearch.storage.segment_writer import SegmentWriter, compress_stored


def flush_memory_segment(mem: MemorySegment, directory: Path, name: str, codec: str) -> SegmentMeta:
    """Write the in-memory buffer as an on-disk segment (deleted docs are kept as tombstones)."""
    writer = SegmentWriter(directory, name, codec)
    for term, docs, tfs, positions, min_dl in mem.iter_for_flush():
        writer.add_term(term, docs, tfs, positions, min_dl)
    blobs = [compress_stored(d) for d in mem.stored_docs]
    meta = writer.finish(mem._lengths_array(), blobs, list(mem.ids))
    return meta


def merge_segments(
    segments: Sequence[DiskSegment], directory: Path, name: str, codec: str
) -> SegmentMeta:
    """K-way merge of segments into one, dropping deleted documents.

    1. For every input segment, build ``remap[local] -> new local ID`` (or -1
       if deleted). New IDs are assigned in input order, so they stay sorted.
    2. Walk all term dictionaries at once with a heap (k-way merge of sorted
       streams). For each term, decode its postings in each segment that has
       it, remap and filter the doc IDs, and concatenate. Order is preserved,
       so no sorting is needed.
    3. Copy the doc store. Stored-field blobs are copied as compressed bytes,
       without recompressing.
    """
    remaps: list[np.ndarray] = []
    next_id = 0
    for seg in segments:
        live = np.ones(seg.max_doc, dtype=bool)
        if seg.deleted_local:
            live[list(seg.deleted_local)] = False
        remap = np.full(seg.max_doc, -1, dtype=np.int64)
        remap[live] = np.arange(next_id, next_id + int(live.sum()))
        next_id += int(live.sum())
        remaps.append(remap)

    new_lengths = (
        np.concatenate(
            [seg.doc_lengths[remap >= 0] for seg, remap in zip(segments, remaps, strict=True)]
        )
        if segments
        else np.empty(0, dtype=np.int64)
    )

    writer = SegmentWriter(directory, name, codec)

    def stream(i: int, seg: DiskSegment):
        # A helper function binds ``i`` eagerly. A nested generator expression
        # would capture it late, so every stream would report the last index.
        for term, entry in seg.files.iter_entries():
            yield term, i, entry

    streams = [stream(i, seg) for i, seg in enumerate(segments)]
    current: str | None = None
    parts: list[tuple[int, tuple]] = []

    def emit(term: str, parts: list[tuple[int, tuple]]) -> None:
        all_docs, all_tfs, all_pos = [], [], []
        for i, entry in parts:
            seg = segments[i]
            docs, tfs = seg.files.decode_postings(entry)
            positions = seg.files.decode_positions(entry, tfs)
            new_docs = remaps[i][docs]
            keep = new_docs >= 0
            if not keep.all():
                pos_keep = np.repeat(keep, tfs)
                positions = positions[pos_keep]
                new_docs, tfs = new_docs[keep], tfs[keep]
            if len(new_docs):
                all_docs.append(new_docs)
                all_tfs.append(tfs)
                all_pos.append(positions)
        if not all_docs:
            return  # every posting of this term was deleted
        docs = np.concatenate(all_docs)
        tfs = np.concatenate(all_tfs)
        min_dl = int(new_lengths[docs].min())
        writer.add_term(term, docs, tfs, np.concatenate(all_pos), min_dl)

    for term, i, entry in heapq.merge(*streams, key=lambda x: (x[0], x[1])):
        if term != current:
            if current is not None:
                emit(current, parts)
            current, parts = term, []
        parts.append((i, entry))
    if current is not None:
        emit(current, parts)

    blobs: list[bytes] = []
    ids: list[str] = []
    for seg, remap in zip(segments, remaps, strict=True):
        for local in np.flatnonzero(remap >= 0):
            blobs.append(bytes(seg.files.stored_blob(int(local))))
            ids.append(seg.files.ids[int(local)])
    return writer.finish(new_lengths, blobs, ids)


@dataclass
class LogMergePolicy:
    """Merge ``merge_factor`` adjacent segments of the same size tier."""

    merge_factor: int = 10
    min_segment_docs: int = 1000
    max_deleted_ratio: float = 0.3

    def tier(self, meta: SegmentMeta) -> int:
        size = max(meta.num_live, 1)
        return (
            max(0, int(math.log(size / self.min_segment_docs, self.merge_factor)))
            if size > self.min_segment_docs
            else 0
        )

    def find_merge(self, metas: Sequence[SegmentMeta]) -> tuple[int, int] | None:
        """Return the half-open index range ``[start, end)`` of segments to merge, or None."""
        tiers = [self.tier(m) for m in metas]
        f = self.merge_factor
        # Prefer the lowest tier: small merges are cheap and cut the segment count most.
        for t in sorted(set(tiers)):
            run = 0
            for i, ti in enumerate(tiers):
                run = run + 1 if ti == t else 0
                if run == f:
                    return i - f + 1, i + 1
        for i, m in enumerate(metas):
            if m.max_doc and m.num_deleted / m.max_doc > self.max_deleted_ratio:
                return i, i + 1
        return None
