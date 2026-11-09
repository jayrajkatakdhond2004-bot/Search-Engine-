"""Read side of an on-disk segment.

Two classes, so that searcher snapshots are truly point-in-time:

* :class:`SegmentFiles` owns the memory-mapped files, the sparse term index
  and the caches. One per segment, shared and never mutated.
* :class:`DiskSegment` is a cheap *view*: ``(files, doc_base, deleted)``.
  A delete or a merge elsewhere in the index creates a **new** view
  (copy-on-write) instead of mutating one a searcher may be using.
"""

from __future__ import annotations

import bisect
import json
import mmap
import struct
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from minisearch.indexer.segment import Postings, Segment, TermInfo
from minisearch.storage.codecs import IntCodec, delta_decode_runs, vbyte_decode_py
from minisearch.storage.format import (
    BLOCK_SIZE,
    CRC,
    HEADER_SIZE,
    KIND_DICT,
    KIND_DOCS,
    KIND_POS,
    KIND_POST,
    SKIP,
    CorruptIndexError,
    SegmentMeta,
    check_header,
    verify_crc,
)
from minisearch.storage.segment_writer import decompress_stored

# (df, cf, max_tf, min_dl, post_off, post_len, pos_off, pos_len)
TermEntry = tuple[int, int, int, int, int, int, int, int]


def _read_vbyte(buf: memoryview | bytes, pos: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        if b < 0x80:
            return v, pos
        shift += 7


def _mmap(path: Path) -> mmap.mmap | bytes:
    with open(path, "rb") as f:
        if path.stat().st_size == 0:
            return b""
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


class _LRU(OrderedDict):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity

    def get_or(self, key, factory):
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = factory()
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)
        return value


class SegmentFiles:
    """Memory-mapped files of one segment plus the in-memory sparse term index."""

    def __init__(self, directory: Path, meta: SegmentMeta, verify: bool = False) -> None:
        self.directory = Path(directory)
        self.meta = meta
        name = meta.name
        self._dict = _mmap(self.directory / f"{name}.dict")
        self._post = _mmap(self.directory / f"{name}.post")
        self._pos = _mmap(self.directory / f"{name}.pos")
        self._docs = _mmap(self.directory / f"{name}.docs")
        codec_id = check_header(self._post, KIND_POST)
        check_header(self._pos, KIND_POS)
        check_header(self._dict, KIND_DICT)
        check_header(self._docs, KIND_DOCS)
        self.codec = IntCodec.from_id(codec_id)
        if verify:
            self.verify()

        # Sparse dictionary index: first term of every block of DICT_BLOCK terms.
        footer_at = len(self._dict) - CRC.size - 16
        index_offset, self.num_terms = struct.unpack_from("<QQ", self._dict, footer_at)
        view = memoryview(self._dict)
        p = HEADER_SIZE + index_offset
        nblocks, p = _read_vbyte(view, p)
        self._block_first: list[str] = []
        self._block_offset: list[int] = []
        for _ in range(nblocks):
            ln, p = _read_vbyte(view, p)
            self._block_first.append(bytes(view[p : p + ln]).decode("utf-8"))
            p += ln
            off, p = _read_vbyte(view, p)
            self._block_offset.append(off)

        # Doc store: lengths and offsets are numpy views straight onto the mmap.
        (n,) = struct.unpack_from("<Q", self._docs, HEADER_SIZE)
        base = HEADER_SIZE + 8
        self.doc_lengths = np.frombuffer(self._docs, dtype="<u4", count=n, offset=base)
        self._doc_offsets = np.frombuffer(self._docs, dtype="<u8", count=n + 1, offset=base + 4 * n)
        self._blob_base = base + 4 * n + 8 * (n + 1)

        with open(self.directory / f"{name}.ids", encoding="utf-8") as fh:
            self.ids: list[str] = json.load(fh)

        self._block_cache = _LRU(4096)
        self._terms: list[str] | None = None

    # ---------------------------------------------------------------- checks

    def verify(self) -> None:
        """Verify CRC32 checksums of every file (slow for big segments)."""
        for buf in (self._dict, self._post, self._pos, self._docs):
            verify_crc(buf)

    # ------------------------------------------------------------ dictionary

    def _decode_block(self, b: int) -> tuple[list[str], list[TermEntry]]:
        def build():
            view = memoryview(self._dict)
            p = HEADER_SIZE + self._block_offset[b]
            count, p = _read_vbyte(view, p)
            post_off, p = _read_vbyte(view, p)
            pos_off, p = _read_vbyte(view, p)
            terms: list[str] = []
            entries: list[TermEntry] = []
            prev = b""
            for _ in range(count):
                shared, p = _read_vbyte(view, p)
                slen, p = _read_vbyte(view, p)
                term = prev[:shared] + bytes(view[p : p + slen])
                p += slen
                vals = []
                for _ in range(6):
                    v, p = _read_vbyte(view, p)
                    vals.append(v)
                df, cf_minus_df, max_tf, min_dl, post_len, pos_len = vals
                entries.append(
                    (df, cf_minus_df + df, max_tf, min_dl, post_off, post_len, pos_off, pos_len)
                )
                post_off += post_len
                pos_off += pos_len
                terms.append(term.decode("utf-8"))
                prev = term
            return terms, entries

        return self._block_cache.get_or(b, build)

    def lookup(self, term: str) -> TermEntry | None:
        b = bisect.bisect_right(self._block_first, term) - 1
        if b < 0:
            return None
        terms, entries = self._decode_block(b)
        i = bisect.bisect_left(terms, term)
        if i < len(terms) and terms[i] == term:
            return entries[i]
        return None

    def iter_entries(self, start_block: int = 0) -> Iterator[tuple[str, TermEntry]]:
        """Iterate ``(term, entry)`` in sorted order, decoding blocks lazily."""
        for b in range(start_block, len(self._block_offset)):
            terms, entries = self._decode_block(b)
            yield from zip(terms, entries, strict=True)

    def terms(self) -> list[str]:
        if self._terms is None:
            self._terms = [t for t, _ in self.iter_entries()]
        return self._terms

    def terms_with_prefix(self, prefix: str) -> list[str]:
        b = max(bisect.bisect_right(self._block_first, prefix) - 1, 0)
        out = []
        for term, _ in self.iter_entries(b):
            if term.startswith(prefix):
                out.append(term)
            elif term > prefix:
                break
        return out

    # -------------------------------------------------------------- postings

    def skip_table(self, entry: TermEntry) -> np.ndarray | None:
        """``(nblocks, 3)`` array of ``(last_doc, post_rel, pos_rel)`` or None for 1 block."""
        df, _, _, _, post_off, _, _, _ = entry
        nblocks = (df + BLOCK_SIZE - 1) // BLOCK_SIZE
        if nblocks <= 1:
            return None
        return np.frombuffer(
            self._post, dtype="<u4", count=3 * nblocks, offset=HEADER_SIZE + post_off
        ).reshape(nblocks, 3)

    def decode_block(self, entry: TermEntry, block: int) -> tuple[np.ndarray, np.ndarray]:
        """Decode one postings block into (local doc IDs, tfs). Used by skip-pointer iterators."""
        df, _, _, _, post_off, post_len, _, _ = entry
        nblocks = (df + BLOCK_SIZE - 1) // BLOCK_SIZE
        skip = self.skip_table(entry)
        data_start = HEADER_SIZE + post_off + (SKIP.size * nblocks if nblocks > 1 else 0)
        if skip is None:
            start, end = data_start, HEADER_SIZE + post_off + post_len
            base = 0
        else:
            start = data_start + int(skip[block, 1])
            end = (
                data_start + int(skip[block + 1, 1])
                if block + 1 < nblocks
                else HEADER_SIZE + post_off + post_len
            )
            base = int(skip[block - 1, 0]) if block > 0 else 0
        n = min(BLOCK_SIZE, df - block * BLOCK_SIZE)
        vals = self.codec.decode(memoryview(self._post)[start:end]).astype(np.int64)
        docs, tfs = vals[:n], vals[n : 2 * n]
        if self.codec.uses_gaps:
            docs = np.cumsum(docs)
            if block > 0:
                docs += base
        return docs, tfs

    def decode_postings(self, entry: TermEntry) -> tuple[np.ndarray, np.ndarray]:
        """Decode a whole term's postings into (local doc IDs, tfs) in one vectorised pass."""
        df, _, _, _, post_off, post_len, _, _ = entry
        nblocks = (df + BLOCK_SIZE - 1) // BLOCK_SIZE
        skip_bytes = SKIP.size * nblocks if nblocks > 1 else 0
        start = HEADER_SIZE + post_off + skip_bytes
        end = HEADER_SIZE + post_off + post_len
        raw = memoryview(self._post)[start:end]
        if df < 8 and self.codec.uses_gaps:
            vals = vbyte_decode_py(raw)
            docs = np.cumsum(np.asarray(vals[:df], dtype=np.int64))
            return docs, np.asarray(vals[df:], dtype=np.int64)
        vals = self.codec.decode(raw).astype(np.int64)
        nfull = df // BLOCK_SIZE
        rem = df - nfull * BLOCK_SIZE
        if nfull:
            full = vals[: 2 * BLOCK_SIZE * nfull].reshape(nfull, 2, BLOCK_SIZE)
            rest = vals[2 * BLOCK_SIZE * nfull :]
            docs = np.concatenate((full[:, 0, :].ravel(), rest[:rem]))
            tfs = np.concatenate((full[:, 1, :].ravel(), rest[rem:]))
        else:
            docs, tfs = vals[:df], vals[df:]
        if self.codec.uses_gaps:
            docs = np.cumsum(docs)
        return docs, tfs

    def decode_positions(self, entry: TermEntry, tfs: np.ndarray) -> np.ndarray:
        """Decode all positions of a term (flat; split them with ``tfs``)."""
        _, _, _, _, _, _, pos_off, pos_len = entry
        raw = memoryview(self._pos)[HEADER_SIZE + pos_off : HEADER_SIZE + pos_off + pos_len]
        vals = self.codec.decode(raw).astype(np.int64)
        if self.codec.uses_gaps:
            vals = delta_decode_runs(vals, tfs)
        return vals

    # ------------------------------------------------------------- doc store

    def stored_blob(self, local: int) -> memoryview:
        s = self._blob_base + int(self._doc_offsets[local])
        e = self._blob_base + int(self._doc_offsets[local + 1])
        return memoryview(self._docs)[s:e]

    def stored(self, local: int) -> dict:
        return decompress_stored(self.stored_blob(local))

    def close(self) -> None:
        for m in (self._dict, self._post, self._pos, self._docs):
            if isinstance(m, mmap.mmap):
                try:
                    m.close()
                except BufferError:
                    # numpy views still reference the map; the OS reclaims it on GC.
                    pass


class DiskSegment(Segment):
    """Immutable view of a :class:`SegmentFiles` at a given doc base with a deletion set."""

    def __init__(self, files: SegmentFiles, doc_base: int, deleted_local: frozenset[int]) -> None:
        self.files = files
        self.doc_base = doc_base
        self._deleted_local = deleted_local
        self._deleted_global: set[int] | None = None

    @property
    def name(self) -> str:
        return self.files.meta.name

    @property
    def max_doc(self) -> int:
        return len(self.files.doc_lengths)

    @property
    def doc_lengths(self) -> np.ndarray:
        return self.files.doc_lengths

    @property
    def deleted_local(self) -> frozenset[int]:
        return self._deleted_local

    @property
    def deleted(self) -> set[int]:
        if self._deleted_global is None:
            self._deleted_global = {self.doc_base + d for d in self._deleted_local}
        return self._deleted_global

    @property
    def total_length(self) -> int:
        total = self.files.meta.total_length
        for d in self._deleted_local:
            total -= int(self.files.doc_lengths[d])
        return total

    def with_changes(
        self, doc_base: int | None = None, deleted_local: frozenset[int] | None = None
    ) -> DiskSegment:
        return DiskSegment(
            self.files,
            self.doc_base if doc_base is None else doc_base,
            self._deleted_local if deleted_local is None else deleted_local,
        )

    def term_info(self, term: str) -> TermInfo | None:
        e = self.files.lookup(term)
        if e is None:
            return None
        return TermInfo(df=e[0], cf=e[1], max_tf=e[2], min_dl=e[3])

    def entry(self, term: str) -> TermEntry | None:
        return self.files.lookup(term)

    def postings(self, term: str) -> Postings | None:
        e = self.files.lookup(term)
        if e is None:
            return None
        docs, tfs = self.files.decode_postings(e)
        files = self.files
        return Postings(
            docs + self.doc_base, tfs, flat_loader=lambda: files.decode_positions(e, tfs)
        )

    def terms(self) -> list[str]:
        return self.files.terms()

    def terms_with_prefix(self, prefix: str) -> list[str]:
        return self.files.terms_with_prefix(prefix)

    def stored(self, doc_id: int) -> dict:
        return self.files.stored(doc_id - self.doc_base)

    def external_id(self, local: int) -> str:
        return self.files.ids[local]


__all__ = ["CorruptIndexError", "DiskSegment", "SegmentFiles", "TermEntry"]
