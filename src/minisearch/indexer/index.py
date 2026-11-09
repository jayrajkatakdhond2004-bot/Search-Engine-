"""The :class:`Index` facade: incremental indexing over segments.

Lifecycle of a document::

    add() -> analysed -> inverted into the in-memory buffer (searchable at once)
          -> flush (buffer full, or commit()) -> immutable on-disk segment
          -> merged with neighbours by the merge policy

Updates are delete + re-add. A delete marks a tombstone in the doc's
segment; the space comes back when that segment is merged. Readers get
point-in-time snapshots (:meth:`Index.searcher`): writes made after a
snapshot never change what that snapshot sees.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from minisearch.analysis import Analyzer, AnalyzerConfig
from minisearch.indexer.memory import MemorySegment
from minisearch.indexer.merge import LogMergePolicy, flush_memory_segment, merge_segments
from minisearch.query.searcher import IndexSearcher
from minisearch.storage.disk_segment import DiskSegment, SegmentFiles
from minisearch.storage.format import SegmentMeta
from minisearch.storage.manifest import Manifest, deletes_file

# Gap between title and body positions, so a phrase can never match across the field boundary.
FIELD_POSITION_GAP = 100


class Index:
    """A persistent, incrementally updatable inverted index stored in ``path``."""

    def __init__(
        self,
        path: str | Path,
        analyzer_config: AnalyzerConfig | None = None,
        codec: str = "vbyte",
        max_buffered_docs: int = 10_000,
        merge_policy: LogMergePolicy | None = None,
        auto_merge: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_buffered_docs = max_buffered_docs
        self.merge_policy = merge_policy or LogMergePolicy()
        self.auto_merge = auto_merge
        self._lock = threading.RLock()

        if Manifest.exists(self.path):
            self.manifest = Manifest.load(self.path)
            stored_cfg = AnalyzerConfig.from_dict(self.manifest.analyzer)
            if analyzer_config is not None and analyzer_config != stored_cfg:
                raise ValueError(
                    "index was built with a different analyzer; "
                    "queries must be analysed exactly like documents"
                )
            analyzer_config = stored_cfg
        else:
            analyzer_config = analyzer_config or AnalyzerConfig()
            self.manifest = Manifest(analyzer=analyzer_config.to_dict(), codec=codec)
            self.manifest.save(self.path)
        self.analyzer = Analyzer(analyzer_config)

        self._files: dict[str, SegmentFiles] = {}
        self._deleted: dict[str, frozenset[int]] = {}
        self._dirty_deletes: set[str] = set()
        self._id_map: dict[str, tuple[str | None, int]] = {}
        for meta in self.manifest.segments:
            self._open_segment(meta)
        self._buffer = MemorySegment()
        self._version = 0
        self._cached_searcher: tuple[int, IndexSearcher] | None = None
        self._collect_garbage()

    # ============================================================== opening

    def _open_segment(self, meta: SegmentMeta) -> None:
        files = SegmentFiles(self.path, meta)
        self._files[meta.name] = files
        deleted: frozenset[int] = frozenset()
        if meta.del_gen:
            bits = np.fromfile(self.path / deletes_file(meta.name, meta.del_gen), dtype=np.uint8)
            deleted = frozenset(int(x) for x in np.flatnonzero(np.unpackbits(bits)[: meta.max_doc]))
        self._deleted[meta.name] = deleted
        for local, ext in enumerate(files.ids):
            if local not in deleted:
                self._id_map[ext] = (meta.name, local)

    def _collect_garbage(self) -> None:
        """Delete files no manifest references (left by interrupted flushes/merges)."""
        keep = self.manifest.referenced_files()
        for f in self.path.iterdir():
            if f.is_file() and f.name not in keep:
                try:
                    f.unlink()
                except OSError:  # pragma: no cover - still mapped on Windows
                    pass

    # ============================================================== writing

    def _analyze_doc(self, title: str, body: str) -> list[tuple[str, int]]:
        analyzed = self.analyzer.analyze(title) if title else []
        if body:
            offset = (len(self.analyzer.raw_tokens(title)) if title else 0) + FIELD_POSITION_GAP
            analyzed.extend(self.analyzer.analyze(body, offset))
        return analyzed

    def add(self, doc: dict) -> str:
        """Index a document ``{"id", "title", "url", "body"}`` and return its id.

        Adding an existing id replaces the old version (delete + add).
        """
        ext_id = str(doc.get("id") or uuid.uuid4().hex)
        title = doc.get("title") or ""
        body = doc.get("body") or doc.get("text") or ""
        stored = {"id": ext_id, "title": title, "url": doc.get("url") or "", "body": body}
        analyzed = self._analyze_doc(title, body)
        with self._lock:
            self._delete_locked(ext_id)
            gid = self._buffer.add(analyzed, stored, ext_id)
            self._id_map[ext_id] = (None, gid - self._buffer.doc_base)
            self._version += 1
            if len(self._buffer) >= self.max_buffered_docs:
                self._flush_locked()
        return ext_id

    def add_many(self, docs: Iterable[dict]) -> int:
        n = 0
        for doc in docs:
            self.add(doc)
            n += 1
        return n

    def delete(self, ext_id: str) -> bool:
        """Delete a document by id. Returns False if it does not exist."""
        with self._lock:
            found = self._delete_locked(ext_id)
            if found:
                self._version += 1
            return found

    def _delete_locked(self, ext_id: str) -> bool:
        loc = self._id_map.pop(ext_id, None)
        if loc is None:
            return False
        seg_name, local = loc
        if seg_name is None:
            self._buffer.delete_local(local)
        else:
            self._deleted[seg_name] = self._deleted[seg_name] | {local}
            self._dirty_deletes.add(seg_name)
        return True

    def commit(self) -> None:
        """Flush buffered docs and deletes, run merges, and atomically publish the manifest."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if len(self._buffer):
            name = self.manifest.new_segment_name()
            meta = flush_memory_segment(self._buffer, self.path, name, self.manifest.codec)
            self.manifest.segments.append(meta)
            self._files[name] = SegmentFiles(self.path, meta)
            self._deleted[name] = frozenset(self._buffer.deleted_local)
            if self._buffer.deleted_local:
                self._dirty_deletes.add(name)
            for ext, (seg, local) in list(self._id_map.items()):
                if seg is None:
                    self._id_map[ext] = (name, local)
            self._buffer = MemorySegment()

        self._write_deletes()
        if self.auto_merge:
            while (rng := self.merge_policy.find_merge(self.manifest.segments)) is not None:
                self._merge_range(*rng)
        self.manifest.save(self.path)
        self._collect_garbage()
        self._version += 1

    def _write_deletes(self) -> None:
        by_name = {m.name: m for m in self.manifest.segments}
        for name in self._dirty_deletes:
            meta = by_name.get(name)
            if meta is None:
                continue
            bits = np.zeros(meta.max_doc, dtype=np.uint8)
            if self._deleted[name]:
                bits[list(self._deleted[name])] = 1
            meta.del_gen += 1
            meta.num_deleted = len(self._deleted[name])
            np.packbits(bits).tofile(self.path / deletes_file(name, meta.del_gen))
        self._dirty_deletes.clear()

    def _merge_range(self, start: int, end: int) -> bool:
        """Merge segments ``[start, end)``. Returns False if every doc was deleted."""
        metas = self.manifest.segments[start:end]
        views = [DiskSegment(self._files[m.name], 0, self._deleted[m.name]) for m in metas]
        name = self.manifest.new_segment_name()
        merged = merge_segments(views, self.path, name, self.manifest.codec)
        for m in metas:
            self._files.pop(m.name)
            self._deleted.pop(m.name)
        if merged.max_doc == 0:
            self.manifest.segments[start:end] = []
            return False
        self.manifest.segments[start:end] = [merged]
        files = SegmentFiles(self.path, merged)
        self._files[name] = files
        self._deleted[name] = frozenset()
        for local, ext in enumerate(files.ids):
            self._id_map[ext] = (name, local)
        return True

    def force_merge(self, max_segments: int = 1) -> None:
        """Merge down to at most ``max_segments`` segments (expunging all deletes)."""
        with self._lock:
            self._flush_locked()
            segs = self.manifest.segments
            if len(segs) > max_segments or any(m.num_deleted for m in segs):
                per = max(1, -(-len(segs) // max_segments))
                i = 0
                while i < len(self.manifest.segments):
                    if self._merge_range(i, min(i + per, len(self.manifest.segments))):
                        i += 1
            self.manifest.save(self.path)
            self._collect_garbage()
            self._version += 1

    # ============================================================== reading

    def searcher(self) -> IndexSearcher:
        """Point-in-time snapshot over all segments plus the in-memory buffer."""
        with self._lock:
            if self._cached_searcher and self._cached_searcher[0] == self._version:
                return self._cached_searcher[1]
            segments = []
            base = 0
            for meta in self.manifest.segments:
                segments.append(DiskSegment(self._files[meta.name], base, self._deleted[meta.name]))
                base += meta.max_doc
            if len(self._buffer):
                segments.append(self._buffer.snapshot(doc_base=base))
                self._buffer.doc_base = base
            s = IndexSearcher(segments, self.analyzer)
            self._cached_searcher = (self._version, s)
            return s

    def __contains__(self, ext_id: str) -> bool:
        return ext_id in self._id_map

    def __len__(self) -> int:
        return len(self._id_map)

    def stats(self) -> dict:
        """Document/term counts and on-disk sizes."""
        with self._lock:
            segs = self.manifest.segments
            index_bytes = sum(m.index_bytes for m in segs)
            raw = sum(m.raw_postings_bytes for m in segs)
            postings_bytes = sum(
                m.file_sizes.get("post", 0) + m.file_sizes.get("pos", 0) for m in segs
            )
            return {
                "documents": len(self._id_map),
                "buffered_documents": len(self._buffer),
                "segments": len(segs),
                "terms": self.num_terms(),
                "total_tokens": sum(m.total_length for m in segs) + self._buffer.num_tokens,
                "index_bytes": index_bytes,
                "postings_bytes": postings_bytes,
                "raw_postings_bytes": raw,
                "compression_ratio": round(raw / postings_bytes, 3) if postings_bytes else None,
                "docstore_bytes": sum(m.file_sizes.get("docs", 0) for m in segs),
                "codec": self.manifest.codec,
                "analyzer": self.manifest.analyzer,
                "deleted_documents": sum(m.num_deleted for m in segs),
            }

    def num_terms(self) -> int:
        """Exact number of distinct terms across all segments (k-way merge of dictionaries)."""
        import heapq

        lists = [self._files[m.name].terms() for m in self.manifest.segments]
        if len(self._buffer):
            lists.append(self._buffer.terms())
        if len(lists) == 1:
            return len(lists[0])
        count, prev = 0, None
        for t in heapq.merge(*lists):
            if t != prev:
                count += 1
                prev = t
        return count

    def close(self) -> None:
        with self._lock:
            self.commit()
            for f in self._files.values():
                f.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
