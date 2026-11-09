"""Writes one immutable segment to disk.

Terms must be added in sorted order. Each term's postings are written as:

``.post`` (per term)::

    [skip table: nblocks x (last_doc u32, post_off u32, pos_off u32)]  only if nblocks > 1
    block 0: VByte(doc gaps...) VByte(tfs...)
    block 1: ...

``.pos`` (per term)::

    block 0: VByte(position gaps, restarting at every doc) ...

Doc gaps carry across blocks: block k's first gap is relative to the last doc
of block k-1. Decoding a whole term is then a single prefix sum, while the
skip table still lets a reader jump straight to any block (see
:mod:`minisearch.query.postings_iter`).

``.dict``: front-coded blocks of 16 terms, then a sparse index of each
block's first term, then a footer.
``.docs``: doc lengths, an offset table and zlib-compressed JSON stored fields.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from minisearch.storage.codecs import IntCodec, delta_encode_runs, vbyte_encode_py
from minisearch.storage.format import (
    BLOCK_SIZE,
    CRC,
    DICT_BLOCK,
    KIND_DICT,
    KIND_DOCS,
    KIND_POS,
    KIND_POST,
    SKIP,
    SegmentMeta,
    pack_header,
)


class _ChecksummedFile:
    """Buffered binary writer that tracks the byte offset and a running CRC32."""

    def __init__(self, path: Path, kind: bytes, codec_id: int) -> None:
        self.path = path
        self._f = open(path, "wb", buffering=1 << 20)  # noqa: SIM115 - closed in close()
        self._crc = 0
        self.offset = 0  # payload offset, i.e. bytes written after the header
        self._write_raw(pack_header(kind, codec_id))
        self.offset = 0

    def _write_raw(self, data: bytes) -> None:
        self._f.write(data)
        self._crc = zlib.crc32(data, self._crc)
        self.offset += len(data)

    def write(self, data: bytes) -> None:
        self._write_raw(data)

    def close(self) -> int:
        self._f.write(CRC.pack(self._crc))
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        return self.path.stat().st_size


def _gaps_py(values: list[int]) -> list[int]:
    out = [values[0]]
    out.extend(values[i] - values[i - 1] for i in range(1, len(values)))
    return out


class SegmentWriter:
    """Streams sorted terms into the four segment files."""

    def __init__(self, directory: Path, name: str, codec: str = "vbyte") -> None:
        self.directory = Path(directory)
        self.name = name
        self.codec = IntCodec(codec)
        cid = self.codec.id
        self._post = _ChecksummedFile(self.directory / f"{name}.post", KIND_POST, cid)
        self._pos = _ChecksummedFile(self.directory / f"{name}.pos", KIND_POS, cid)
        self._dict_entries: list[tuple[bytes, tuple[int, ...]]] = []
        self._block_post_start = 0
        self._block_pos_start = 0
        self._last_term: str | None = None
        self.raw_postings_bytes = 0

    # ------------------------------------------------------------------ terms

    def add_term(
        self,
        term: str,
        docs: Sequence[int] | np.ndarray,
        tfs: Sequence[int] | np.ndarray,
        positions: Sequence[int] | np.ndarray,
        min_dl: int = 0,
    ) -> None:
        """Write one term's postings.

        ``docs`` are sorted segment-local doc IDs. ``positions`` is the
        concatenation of each doc's sorted positions; ``tfs[i]`` of them belong
        to ``docs[i]``. ``min_dl`` is the shortest length among ``docs`` (stored
        for WAND score upper bounds).
        """
        if self._last_term is not None and term <= self._last_term:
            raise ValueError(
                f"terms must be added in sorted order: {self._last_term!r} >= {term!r}"
            )
        self._last_term = term
        df = len(docs)
        if df == 0:
            return
        post_off, pos_off = self._post.offset, self._pos.offset
        enc = self.codec.encode
        gaps = self.codec.uses_gaps

        if df <= BLOCK_SIZE:
            # Single block, no skip table. Small lists go through pure Python
            # (numpy's per-call overhead dominates below ~24 values).
            if df < 24:
                d = list(map(int, docs))
                t = list(map(int, tfs))
                p = list(map(int, positions))
                if gaps:
                    pg: list[int] = []
                    i = 0
                    for tf in t:
                        prev = 0
                        for x in p[i : i + tf]:
                            pg.append(x - prev)
                            prev = x
                        i += tf
                    self._post.write(vbyte_encode_py(_gaps_py(d)) + vbyte_encode_py(t))
                    self._pos.write(vbyte_encode_py(pg))
                else:
                    self._post.write(enc(d) + enc(t))
                    self._pos.write(enc(p))
            else:
                d_arr = np.asarray(docs, dtype=np.int64)
                t_arr = np.asarray(tfs, dtype=np.int64)
                p_arr = np.asarray(positions, dtype=np.int64)
                d_enc = np.diff(d_arr, prepend=0) if gaps else d_arr
                p_enc = delta_encode_runs(p_arr, t_arr) if gaps else p_arr
                self._post.write(enc(d_enc) + enc(t_arr))
                self._pos.write(enc(p_enc))
        else:
            self._write_blocked(docs, tfs, positions)

        cf = int(np.sum(tfs)) if isinstance(tfs, np.ndarray) else sum(tfs)
        max_tf = int(np.max(tfs)) if isinstance(tfs, np.ndarray) else max(tfs)
        self.raw_postings_bytes += 4 * (2 * df + cf)
        self._dict_entries.append(
            (
                term.encode("utf-8"),
                (
                    df,
                    cf,
                    max_tf,
                    int(min_dl),
                    self._post.offset - post_off,
                    self._pos.offset - pos_off,
                ),
            )
        )

    def _write_blocked(self, docs, tfs, positions) -> None:
        d_arr = np.asarray(docs, dtype=np.int64)
        t_arr = np.asarray(tfs, dtype=np.int64)
        p_arr = np.asarray(positions, dtype=np.int64)
        gaps = self.codec.uses_gaps
        enc = self.codec.encode
        df = len(d_arr)
        nblocks = (df + BLOCK_SIZE - 1) // BLOCK_SIZE
        d_enc = np.diff(d_arr, prepend=0) if gaps else d_arr
        p_enc = delta_encode_runs(p_arr, t_arr) if gaps else p_arr
        pos_bounds = np.concatenate(([0], np.cumsum(t_arr)))

        skip = bytearray()
        post_blocks: list[bytes] = []
        pos_blocks: list[bytes] = []
        post_rel = pos_rel = 0
        for b in range(nblocks):
            s, e = b * BLOCK_SIZE, min((b + 1) * BLOCK_SIZE, df)
            pb = enc(d_enc[s:e]) + enc(t_arr[s:e])
            qb = enc(p_enc[pos_bounds[s] : pos_bounds[e]])
            skip += SKIP.pack(int(d_arr[e - 1]), post_rel, pos_rel)
            post_blocks.append(pb)
            pos_blocks.append(qb)
            post_rel += len(pb)
            pos_rel += len(qb)
        self._post.write(bytes(skip) + b"".join(post_blocks))
        self._pos.write(b"".join(pos_blocks))

    # ------------------------------------------------------------- dictionary

    def _write_dict(self) -> int:
        f = _ChecksummedFile(self.directory / f"{self.name}.dict", KIND_DICT, self.codec.id)
        index: list[tuple[bytes, int]] = []
        post_off = pos_off = 0
        entries = self._dict_entries
        for start in range(0, len(entries), DICT_BLOCK):
            block = entries[start : start + DICT_BLOCK]
            index.append((block[0][0], f.offset))
            out = bytearray(vbyte_encode_py([len(block), post_off, pos_off]))
            prev = b""
            for term, (df, cf, max_tf, min_dl, post_len, pos_len) in block:
                shared = 0
                limit = min(len(prev), len(term))
                while shared < limit and prev[shared] == term[shared]:
                    shared += 1
                suffix = term[shared:]
                out += vbyte_encode_py([shared, len(suffix)])
                out += suffix
                out += vbyte_encode_py([df, cf - df, max_tf, min_dl, post_len, pos_len])
                post_off += post_len
                pos_off += pos_len
                prev = term
            f.write(bytes(out))
        index_offset = f.offset
        idx = bytearray(vbyte_encode_py([len(index)]))
        for first, off in index:
            idx += vbyte_encode_py([len(first)]) + first + vbyte_encode_py([off])
        f.write(bytes(idx))
        f.write(struct.pack("<QQ", index_offset, len(entries)))
        return f.close()

    # ---------------------------------------------------------------- docs

    def _write_docs(self, doc_lengths: np.ndarray, stored_blobs: Sequence[bytes]) -> int:
        f = _ChecksummedFile(self.directory / f"{self.name}.docs", KIND_DOCS, 0)
        n = len(doc_lengths)
        offsets = np.zeros(n + 1, dtype="<u8")
        if n:
            offsets[1:] = np.cumsum([len(b) for b in stored_blobs])
        f.write(struct.pack("<Q", n))
        f.write(np.asarray(doc_lengths, dtype="<u4").tobytes())
        f.write(offsets.tobytes())
        for blob in stored_blobs:
            f.write(blob)
        return f.close()

    # --------------------------------------------------------------- finish

    def finish(
        self, doc_lengths: Sequence[int] | np.ndarray, stored_blobs: Sequence[bytes], ids: list[str]
    ) -> SegmentMeta:
        """Close all files (fsync'd) and return the segment's metadata."""
        sizes = {
            "post": self._post.close(),
            "pos": self._pos.close(),
            "dict": self._write_dict(),
            "docs": self._write_docs(np.asarray(doc_lengths), stored_blobs),
        }
        ids_path = self.directory / f"{self.name}.ids"
        with open(ids_path, "w", encoding="utf-8") as fh:
            json.dump(ids, fh)
            fh.flush()
            os.fsync(fh.fileno())
        sizes["ids"] = ids_path.stat().st_size
        return SegmentMeta(
            name=self.name,
            max_doc=len(doc_lengths),
            num_terms=len(self._dict_entries),
            total_length=int(np.sum(doc_lengths)) if len(doc_lengths) else 0,
            codec=self.codec.name,
            file_sizes=sizes,
            raw_postings_bytes=self.raw_postings_bytes,
        )


def compress_stored(doc: dict) -> bytes:
    """Stored fields are JSON + zlib, compressed per document for O(1) random access."""
    return zlib.compress(json.dumps(doc, ensure_ascii=False).encode("utf-8"), 6)


def decompress_stored(blob: bytes | memoryview) -> dict:
    return json.loads(zlib.decompress(blob))
