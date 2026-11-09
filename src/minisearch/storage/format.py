"""Shared constants and helpers for the on-disk format (see DESIGN.md).

Every segment file is laid out as::

    +-----------------------------+
    | header (16 bytes)           |  magic "MSRC", file kind (4 bytes), version u16,
    |                             |  codec u8, reserved (5 bytes)
    +-----------------------------+
    | payload                     |
    +-----------------------------+
    | footer                      |  file-specific trailer, ending with a CRC32
    +-----------------------------+  (u32) of all preceding bytes
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

MAGIC = b"MSRC"
VERSION = 1
HEADER = struct.Struct("<4s4sHB5x")  # 16 bytes
HEADER_SIZE = HEADER.size
CRC = struct.Struct("<I")

KIND_DICT = b"DICT"
KIND_POST = b"POST"
KIND_POS = b"POSN"
KIND_DOCS = b"DOCS"

BLOCK_SIZE = 128  # postings per block; one skip entry per block
DICT_BLOCK = 16  # terms per front-coded dictionary block

SKIP = struct.Struct("<III")  # last_doc, postings byte offset, positions byte offset


class CorruptIndexError(Exception):
    """Raised when a checksum, magic number or version check fails."""


def pack_header(kind: bytes, codec_id: int) -> bytes:
    return HEADER.pack(MAGIC, kind, VERSION, codec_id)


def check_header(buf: bytes | memoryview, kind: bytes) -> int:
    """Validate a file header and return its codec id."""
    magic, got_kind, version, codec_id = HEADER.unpack_from(buf, 0)
    if magic != MAGIC or got_kind != kind:
        raise CorruptIndexError(f"bad magic/kind: {magic!r} {got_kind!r}, expected {kind!r}")
    if version != VERSION:
        raise CorruptIndexError(f"unsupported format version {version}")
    return codec_id


def verify_crc(buf: bytes | memoryview) -> None:
    """Check the trailing CRC32 over the whole file."""
    (stored,) = CRC.unpack_from(buf, len(buf) - CRC.size)
    if zlib.crc32(buf[: len(buf) - CRC.size]) != stored:
        raise CorruptIndexError("checksum mismatch")


@dataclass
class SegmentMeta:
    """Per-segment metadata persisted in the manifest."""

    name: str
    max_doc: int
    num_terms: int
    total_length: int
    codec: str
    del_gen: int = 0
    num_deleted: int = 0
    file_sizes: dict[str, int] = field(default_factory=dict)
    raw_postings_bytes: int = 0  # size if doc IDs/tfs/positions were plain uint32

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> SegmentMeta:
        return cls(**d)

    @property
    def index_bytes(self) -> int:
        """Inverted index size: dictionary + postings + positions (excludes doc store)."""
        return sum(v for k, v in self.file_sizes.items() if k in ("dict", "post", "pos"))

    @property
    def num_live(self) -> int:
        return self.max_doc - self.num_deleted
