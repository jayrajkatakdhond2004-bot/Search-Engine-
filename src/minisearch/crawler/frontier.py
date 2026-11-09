"""The crawl frontier: per-host FIFO queues scheduled by politeness, persisted to disk.

Scheduling. Every host has its own FIFO queue (breadth-first within a host)
and a ``next_allowed`` time. A min-heap orders hosts by ``next_allowed``.
``pop()`` returns a URL from the host that becomes available soonest, so
while one host is cooling down the crawler works on another, and no host is
ever hit faster than its delay. (This is the "back queue" half of the
Mercator frontier design.)

Dedup. ``seen`` holds every normalised URL ever enqueued, so a URL is fetched
at most once even when thousands of pages link to it.

Resumability. ``save()`` writes the queues, the seen set and the counters
atomically (temp file + rename). After a crash or Ctrl-C, ``load()``
continues exactly where the crawl stopped.
"""

from __future__ import annotations

import heapq
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from minisearch.crawler.urls import host_of


@dataclass(order=True)
class _HostSlot:
    ready_at: float
    host: str


class Frontier:
    def __init__(self) -> None:
        self.queues: dict[str, deque[tuple[str, int]]] = {}
        self.next_allowed: dict[str, float] = {}
        self.seen: set[str] = set()
        self._heap: list[_HostSlot] = []
        self._in_heap: set[str] = set()

    def __len__(self) -> int:
        return sum(len(q) for q in self.queues.values())

    def add(self, url: str, depth: int) -> bool:
        """Enqueue ``url`` unless it was seen before. Returns True if it was added."""
        if url in self.seen:
            return False
        self.seen.add(url)
        host = host_of(url)
        self.queues.setdefault(host, deque()).append((url, depth))
        self._schedule(host)
        return True

    def _schedule(self, host: str) -> None:
        if host not in self._in_heap and self.queues.get(host):
            heapq.heappush(self._heap, _HostSlot(self.next_allowed.get(host, 0.0), host))
            self._in_heap.add(host)

    def pop(self, now: float | None = None) -> tuple[str, int, float] | None:
        """Return ``(url, depth, wait_seconds)`` for the next URL, or None if empty.

        The caller must sleep ``wait_seconds`` before fetching (politeness).
        """
        while self._heap:
            slot = heapq.heappop(self._heap)
            self._in_heap.discard(slot.host)
            q = self.queues.get(slot.host)
            if not q:
                continue
            url, depth = q.popleft()
            now = time.monotonic() if now is None else now
            wait = max(0.0, self.next_allowed.get(slot.host, 0.0) - now)
            return url, depth, wait
        return None

    def done(self, url: str, delay: float, now: float | None = None) -> None:
        """Record a fetch of ``url``. Its host may not be hit again for ``delay`` seconds."""
        host = host_of(url)
        now = time.monotonic() if now is None else now
        self.next_allowed[host] = now + delay
        self._schedule(host)

    def release(self, url: str) -> None:
        """Return ``url``'s host to the schedule without a fetch (e.g. robots.txt disallowed).

        Every ``pop()`` must be followed by exactly one ``done()`` or ``release()``.
        Otherwise the host drops off the heap and its remaining URLs are stranded.
        """
        self._schedule(host_of(url))

    # ------------------------------------------------------------ persistence

    def save(self, path: Path, extra: dict | None = None) -> None:
        state = {
            "queues": {h: list(q) for h, q in self.queues.items() if q},
            "seen": sorted(self.seen),
            "extra": extra or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> tuple[Frontier, dict]:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        fr = cls()
        fr.seen = set(state["seen"])
        for host, items in state["queues"].items():
            fr.queues[host] = deque((u, d) for u, d in items)
            fr._schedule(host)
        return fr, state.get("extra", {})
