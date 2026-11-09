"""The manifest: the single source of truth for which segments make up the index.

A commit is atomic. Segment files are written and fsync'd first. Then
``manifest.json`` is replaced with :func:`os.replace` (an atomic rename on
POSIX and Windows), and the directory is fsync'd. A crash at any point
leaves either the old manifest or the new one, never a mix. Files no
manifest references are orphans from an interrupted flush or merge, and are
deleted on open.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from minisearch.storage.format import SegmentMeta

MANIFEST = "manifest.json"
FORMAT_VERSION = 1


@dataclass
class Manifest:
    analyzer: dict
    codec: str = "vbyte"
    generation: int = 0
    next_segment: int = 0
    segments: list[SegmentMeta] = field(default_factory=list)
    version: int = FORMAT_VERSION

    def new_segment_name(self) -> str:
        name = f"seg_{self.next_segment:06d}"
        self.next_segment += 1
        return name

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "generation": self.generation,
            "codec": self.codec,
            "analyzer": self.analyzer,
            "next_segment": self.next_segment,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        if d.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported manifest version {d.get('version')}")
        return cls(
            analyzer=d["analyzer"],
            codec=d["codec"],
            generation=d["generation"],
            next_segment=d["next_segment"],
            segments=[SegmentMeta.from_dict(s) for s in d["segments"]],
        )

    # ------------------------------------------------------------------ disk

    @staticmethod
    def exists(directory: Path) -> bool:
        return (Path(directory) / MANIFEST).exists()

    @classmethod
    def load(cls, directory: Path) -> Manifest:
        with open(Path(directory) / MANIFEST, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, directory: Path) -> None:
        """Atomically write the manifest (write temp, fsync, rename, fsync dir)."""
        directory = Path(directory)
        self.generation += 1
        tmp = directory / f"{MANIFEST}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, directory / MANIFEST)
        _fsync_dir(directory)

    def referenced_files(self) -> set[str]:
        files = {MANIFEST}
        for s in self.segments:
            files.update(f"{s.name}.{ext}" for ext in ("dict", "post", "pos", "docs", "ids"))
            if s.del_gen:
                files.add(deletes_file(s.name, s.del_gen))
        return files


def deletes_file(segment: str, gen: int) -> str:
    return f"{segment}_{gen}.del"


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - Windows cannot open directories
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
