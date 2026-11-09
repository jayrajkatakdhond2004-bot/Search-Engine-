"""Integer compression codecs: delta (d-gap) encoding + variable-byte (VByte).

Why this combination
--------------------
Postings store sorted doc IDs. Storing the gaps between them ("d-gaps")
instead of absolute values turns most numbers small: frequent terms have
gaps near 1. VByte then stores each integer in 7-bit groups, using the high
bit of each byte as a continuation flag:

    value 5     -> 00000101                       (1 byte)
    value 300   -> 10101100 00000010              (2 bytes, low group first)

VByte is byte-aligned, so decoding needs no bit shuffling. Bit-level codes
(Elias-gamma, Golomb-Rice) compress better, and SIMD codes (PForDelta,
Stream-VByte) decode faster, but VByte is the classic sweet spot of
simplicity and speed. See DESIGN.md for the trade-offs.

Two implementations
-------------------
* ``vbyte_encode_py`` / ``vbyte_decode_py``: obviously correct reference loops.
* ``vbyte_encode`` / ``vbyte_decode``: numpy-vectorised versions. In CPython a
  per-byte loop costs about 50-100 ns, so decoding a 1M-posting list would take
  ~0.1 s. The vectorised decoder does it in a few milliseconds.

Tests check that both implementations agree on random inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

_SMALL = 24  # below this many values the pure-Python path beats numpy's call overhead


# --------------------------------------------------------------------------- reference


def vbyte_encode_py(values: Iterable[int]) -> bytes:
    """Encode non-negative integers with VByte (reference implementation)."""
    out = bytearray()
    for v in values:
        if v < 0:
            raise ValueError("VByte encodes non-negative integers only")
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def vbyte_decode_py(data: bytes | memoryview, count: int | None = None) -> list[int]:
    """Decode VByte bytes (reference implementation)."""
    out: list[int] = []
    v = shift = 0
    for byte in bytes(data):
        v |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            out.append(v)
            if count is not None and len(out) == count:
                break
            v = shift = 0
    return out


# --------------------------------------------------------------------------- vectorised


def vbyte_encode(values: Sequence[int] | np.ndarray) -> bytes:
    """Vectorised VByte encoder.

    1. Compute the number of 7-bit groups each value needs.
    2. Compute each value's output byte offset (an exclusive prefix sum).
    3. For each group j, scatter ``(v >> 7j) & 0x7F`` (plus the continuation
       bit) into the output at ``offset + j``, for all values that have a
       j-th group.
    """
    n = len(values)
    if n < _SMALL:
        return vbyte_encode_py(int(v) for v in values)
    arr = np.asarray(values, dtype=np.uint64)
    # nbytes = max(1, ceil(bit_length / 7)), computed by threshold counting.
    nbytes = np.ones(n, dtype=np.int64)
    thr = np.uint64(0x80)
    for _ in range(9):
        nbytes += arr >= thr
        thr = thr << np.uint64(7)
    offsets = np.empty(n, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(nbytes[:-1], out=offsets[1:])
    out = np.empty(int(offsets[-1] + nbytes[-1]), dtype=np.uint8)
    max_groups = int(nbytes.max())
    for j in range(max_groups):
        sel = nbytes > j
        group = (arr[sel] >> np.uint64(7 * j)) & np.uint64(0x7F)
        cont = (nbytes[sel] - 1 > j).astype(np.uint64) << np.uint64(7)
        out[offsets[sel] + j] = (group | cont).astype(np.uint8)
    return out.tobytes()


def vbyte_decode(data: bytes | memoryview | np.ndarray, count: int | None = None) -> np.ndarray:
    """Vectorised VByte decoder. Returns ``uint64`` values.

    A byte with the high bit clear ends a value. ``np.add.reduceat`` sums the
    shifted 7-bit groups of each value. The groups occupy disjoint bits, so
    the sum equals bitwise OR.
    """
    buf = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else data
    if len(buf) == 0:
        return np.empty(0, dtype=np.uint64)
    ends = np.flatnonzero(buf < 0x80)
    if count is not None:
        ends = ends[:count]
        buf = buf[: int(ends[-1]) + 1] if len(ends) else buf[:0]
    if len(ends) == 0:
        return np.empty(0, dtype=np.uint64)
    if len(ends) == len(buf):  # every value fits in one byte: common for gaps and tfs
        return buf.astype(np.uint64)
    starts = np.empty(len(ends), dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    idx_in_group = np.arange(len(buf), dtype=np.int64) - np.repeat(starts, ends - starts + 1)
    shifted = (buf & 0x7F).astype(np.uint64) << (np.uint64(7) * idx_in_group.astype(np.uint64))
    return np.add.reduceat(shifted, starts)


# --------------------------------------------------------------------------- delta


def delta_encode(sorted_values: np.ndarray, base: int = 0) -> np.ndarray:
    """Gaps between consecutive sorted values, the first relative to ``base``."""
    arr = np.asarray(sorted_values, dtype=np.int64)
    if len(arr) == 0:
        return arr
    gaps = np.empty_like(arr)
    gaps[0] = arr[0] - base
    np.subtract(arr[1:], arr[:-1], out=gaps[1:])
    if (gaps < 0).any():
        raise ValueError("delta encoding requires non-decreasing input")
    return gaps


def delta_decode(gaps: np.ndarray, base: int = 0) -> np.ndarray:
    """Inverse of :func:`delta_encode` (a prefix sum)."""
    out = np.cumsum(np.asarray(gaps, dtype=np.int64))
    if base:
        out += base
    return out


def delta_encode_runs(values: np.ndarray, run_lengths: np.ndarray) -> np.ndarray:
    """Delta-encode several concatenated sorted runs, restarting at each run.

    Positions are stored per document, so the gap resets at every document
    boundary: ``[3, 7, 9 | 2, 5]`` (tfs ``[3, 2]``) becomes ``[3, 4, 2 | 2, 3]``.
    """
    vals = np.asarray(values, dtype=np.int64)
    gaps = np.empty_like(vals)
    if len(vals) == 0:
        return gaps
    gaps[0] = vals[0]
    np.subtract(vals[1:], vals[:-1], out=gaps[1:])
    starts = np.cumsum(run_lengths)[:-1]
    gaps[starts] = vals[starts]
    return gaps


def delta_decode_runs(gaps: np.ndarray, run_lengths: np.ndarray) -> np.ndarray:
    """Inverse of :func:`delta_encode_runs`: a prefix sum that restarts at every run."""
    g = np.asarray(gaps, dtype=np.int64)
    if len(g) == 0:
        return g
    cs = np.cumsum(g)
    run_lengths = np.asarray(run_lengths, dtype=np.int64)
    run_starts = np.concatenate(([0], np.cumsum(run_lengths)[:-1]))
    before = np.where(run_starts > 0, cs[np.maximum(run_starts - 1, 0)], 0)
    return cs - np.repeat(before, run_lengths)


# --------------------------------------------------------------------------- codec facade


class IntCodec:
    """Encodes a block of integers. ``vbyte`` (gap + VByte) or ``raw`` (fixed uint32)."""

    def __init__(self, name: str) -> None:
        if name not in ("vbyte", "raw"):
            raise ValueError(f"unknown codec {name!r}")
        self.name = name
        self.id = 1 if name == "vbyte" else 0

    @classmethod
    def from_id(cls, codec_id: int) -> IntCodec:
        return cls("vbyte" if codec_id == 1 else "raw")

    def encode(self, values: Sequence[int] | np.ndarray) -> bytes:
        if self.name == "raw":
            return np.asarray(values, dtype="<u4").tobytes()
        return vbyte_encode(values)

    def decode(self, data: bytes | memoryview, count: int | None = None) -> np.ndarray:
        if self.name == "raw":
            arr = np.frombuffer(data, dtype="<u4")
            return arr[:count] if count is not None else arr
        return vbyte_decode(data, count)

    @property
    def uses_gaps(self) -> bool:
        """The raw baseline stores absolute values, so it can show what d-gaps buy."""
        return self.name == "vbyte"
