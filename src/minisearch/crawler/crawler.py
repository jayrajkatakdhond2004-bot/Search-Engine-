"""A polite, resumable, domain-restricted crawler that writes JSONL documents.

Per URL::

    frontier.pop() -> wait out the host's politeness delay
      -> robots.txt allowed?          (cached per host, Crawl-delay honoured)
      -> fetch with retries/backoff   (HTML only, size-capped)
      -> redirect stayed on-domain?
      -> extract title/text/links     (meta robots noindex/nofollow honoured)
      -> exact-duplicate content?     (SHA-1 of the normalised text)
      -> write {"id","title","url","body"} as one JSONL line
      -> enqueue on-domain links up to max_depth

It is single-threaded on purpose. Politeness (one request per host per
``delay`` seconds) limits throughput far more than CPU does, and a
sequential loop is easy to reason about. With many hosts, an async version
fetching from several hosts at once would be the next step.
"""

from __future__ import annotations

import hashlib
import json
import logging
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from minisearch.crawler.extract import extract
from minisearch.crawler.fetcher import Fetcher
from minisearch.crawler.frontier import Frontier
from minisearch.crawler.robots import RobotsCache
from minisearch.crawler.urls import host_of, in_domains, normalize_url

log = logging.getLogger("minisearch.crawler")


@dataclass
class CrawlConfig:
    seeds: list[str]
    allowed_domains: list[str] | None = None  # default: the seeds' hosts
    max_pages: int = 200
    max_depth: int = 5
    delay: float = 1.0  # min seconds between requests to the same host
    state_dir: Path = Path("crawl_state")
    user_agent: str = "minisearch-bot/0.1 (+educational crawler)"
    timeout: float = 15.0
    max_retries: int = 3
    backoff_base: float = 1.0
    save_every: int = 20
    retry_failed: bool = False  # on resume, re-queue URLs that failed transiently


@dataclass
class CrawlStats:
    fetched: int = 0
    written: int = 0
    errors: int = 0
    retries: int = 0
    robots_blocked: int = 0
    duplicates: int = 0
    skipped_non_html: int = 0
    offsite_redirects: int = 0
    noindex: int = 0
    errors_by_kind: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class Crawler:
    def __init__(
        self,
        config: CrawlConfig,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = config
        self.sleep = sleep
        self.client = client or httpx.Client(
            headers={"User-Agent": config.user_agent},
            timeout=config.timeout,
            follow_redirects=True,
            max_redirects=5,
        )
        self.robots = RobotsCache(self.client, config.user_agent)
        seeds = [u for u in (normalize_url(s) for s in config.seeds) if u]
        if not seeds:
            raise ValueError("no valid seed URLs")
        self.domains = set(config.allowed_domains or [host_of(s).split(":")[0] for s in seeds])
        self.fetcher = Fetcher(
            self.client,
            config.max_retries,
            config.backoff_base,
            sleep=sleep,
            allow_redirect=lambda u: in_domains(u, self.domains),
        )
        self.state_path = Path(config.state_dir) / "frontier.json"
        self.stats = CrawlStats()
        self.content_hashes: set[str] = set()
        self.failed: dict[str, int] = {}  # transient failures (network/5xx): url -> depth
        if self.state_path.exists():
            self.frontier, extra = Frontier.load(self.state_path)
            self.stats = CrawlStats(**extra.get("stats", {}))
            self.content_hashes = set(extra.get("hashes", []))
            self.failed = dict(extra.get("failed", {}))
            if config.retry_failed and self.failed:
                for u, d in self.failed.items():
                    self.frontier.seen.discard(u)
                    self.frontier.add(u, d)
                log.info("re-queued %d failed URLs", len(self.failed))
                self.failed = {}
            log.info(
                "resumed crawl: %d queued, %d seen", len(self.frontier), len(self.frontier.seen)
            )
        else:
            self.frontier = Frontier()
            for s in seeds:
                self.frontier.add(s, 0)
        self._stop = False

    def _save(self) -> None:
        self.frontier.save(
            self.state_path,
            {
                "stats": self.stats.as_dict(),
                "hashes": sorted(self.content_hashes),
                "failed": self.failed,
            },
        )

    def _error(self, kind: str) -> None:
        self.stats.errors += 1
        self.stats.errors_by_kind[kind] = self.stats.errors_by_kind.get(kind, 0) + 1

    def _host_delay(self, url: str) -> float:
        crawl_delay = self.robots.crawl_delay(url)
        return max(self.cfg.delay, crawl_delay or 0.0)

    def run(self, out_path: Path) -> int:
        """Crawl until ``max_pages`` documents are written or the frontier is empty.

        Appends to ``out_path``, so a resumed crawl continues the same file.
        Returns the number of documents written in this run.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written_now = 0
        previous = signal.getsignal(signal.SIGINT)

        def on_sigint(signum, frame):  # finish the current page, save, exit cleanly
            self._stop = True

        try:
            signal.signal(signal.SIGINT, on_sigint)
        except ValueError:  # not in the main thread (e.g. tests)
            previous = None
        try:
            with open(out_path, "a", encoding="utf-8") as out:
                while not self._stop and self.stats.written < self.cfg.max_pages:
                    item = self.frontier.pop()
                    if item is None:
                        break
                    url, depth, wait = item
                    if not self.robots.allowed(url):
                        self.stats.robots_blocked += 1
                        self.frontier.release(url)
                        continue
                    if wait > 0:
                        self.sleep(wait)
                    doc = self._process(url, depth)
                    if doc is not None:
                        out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        out.flush()
                        self.stats.written += 1
                        written_now += 1
                        if self.stats.written % self.cfg.save_every == 0:
                            self._save()
        finally:
            self._save()
            if previous is not None:
                signal.signal(signal.SIGINT, previous)
        return written_now

    def _process(self, url: str, depth: int) -> dict | None:
        res = self.fetcher.fetch(url)
        self.frontier.done(url, self._host_delay(url))
        self.stats.fetched += 1
        self.stats.retries += res.retries
        if res.text is None:
            if res.error == "not html":
                self.stats.skipped_non_html += 1
            elif res.error == "offsite redirect":
                self.stats.offsite_redirects += 1
            else:
                self._error(res.error or "unknown")
                if res.status == 0 or res.status >= 500:  # transient: retry on a later run
                    self.failed[url] = depth
            return None

        final = normalize_url(res.url) or url
        if final != url:  # on-domain redirect (checked by the fetcher)
            if final in self.frontier.seen and final != url:
                self.stats.duplicates += 1  # redirect target is crawled separately
                return None
            self.frontier.seen.add(final)

        page = extract(res.text, final)
        if not page.nofollow and depth < self.cfg.max_depth:
            for link in page.links:
                if in_domains(link, self.domains):
                    self.frontier.add(link, depth + 1)
        if page.noindex:
            self.stats.noindex += 1
            return None
        digest = hashlib.sha1(" ".join(page.text.split()).encode("utf-8")).hexdigest()
        if digest in self.content_hashes:
            self.stats.duplicates += 1
            return None
        self.content_hashes.add(digest)
        return {
            "id": page.canonical or final,
            "title": page.title,
            "url": final,
            "body": page.text,
            "depth": depth,
            "crawled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
