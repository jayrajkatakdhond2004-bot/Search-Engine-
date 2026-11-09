"""HTTP fetching with retries, exponential backoff, and size and type limits."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx

from minisearch.crawler.urls import normalize_url

RETRY_STATUS = {429, 500, 502, 503, 504}


def no_downgrade(current: str, target: str) -> str:
    """Rewrite a same-host https -> http redirect to https (what HSTS browsers do).

    Some sites redirect ``https://host/x`` to ``http://host/x/``. Following
    that downgrade would leak the request in clear text.
    """
    cur, tgt = urlsplit(current), urlsplit(target)
    if cur.scheme == "https" and tgt.scheme == "http" and cur.hostname == tgt.hostname:
        port = f":{tgt.port}" if tgt.port not in (None, 80, 443) else ""
        return urlunsplit(("https", f"{tgt.hostname}{port}", tgt.path, tgt.query, ""))
    return target


@dataclass
class FetchResult:
    url: str  # final URL after redirects
    status: int
    content_type: str
    text: str | None  # None when skipped (non-HTML, too large) or failed
    error: str | None = None
    retries: int = 0


class Fetcher:
    """Fetches HTML pages.

    * retries transient failures (connection errors, 429, 5xx) with exponential
      backoff ``base * 2^attempt``, honouring ``Retry-After`` (capped)
    * never follows a same-host https -> http downgrade (it is upgraded back)
    * follows up to 5 redirects *manually*: each ``Location`` is checked with
      ``allow_redirect`` before it is requested, so a redirect can never make the
      crawler leave its allowed domains
    * streams the body and aborts above ``max_bytes``
    * skips non-HTML responses without downloading their body
    """

    def __init__(
        self,
        client: httpx.Client,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        max_bytes: int = 2_000_000,
        sleep: Callable[[float], None] = time.sleep,
        allow_redirect: Callable[[str], bool] = lambda url: True,
        max_redirects: int = 5,
    ) -> None:
        self.client = client
        self.allow_redirect = allow_redirect
        self.max_redirects = max_redirects
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_bytes = max_bytes
        self.sleep = sleep

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        delay = self.backoff_base * (2**attempt)
        if retry_after and retry_after.strip().isdigit():
            delay = max(delay, float(retry_after))
        return min(delay, 60.0)

    def fetch(self, url: str) -> FetchResult:
        retries = 0
        for _ in range(self.max_redirects + 1):
            res, location = self._fetch_one(url)
            retries += res.retries
            res.retries = retries
            if location is None:
                return res
            target = normalize_url(location, url)
            if target is None:
                return FetchResult(url, res.status, "", None, "bad redirect", retries)
            target = no_downgrade(url, target)
            if not self.allow_redirect(target):
                return FetchResult(target, res.status, "", None, "offsite redirect", retries)
            url = target
        return FetchResult(url, 0, "", None, "too many redirects", retries)

    def _fetch_one(self, url: str) -> tuple[FetchResult, str | None]:
        """One URL with retries. Returns ``(result, redirect_location_or_None)``."""
        attempt = 0
        while True:
            try:
                with self.client.stream("GET", url, follow_redirects=False) as resp:
                    status = resp.status_code
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if status in RETRY_STATUS and attempt < self.max_retries:
                        self.sleep(self._backoff(attempt, resp.headers.get("retry-after")))
                        attempt += 1
                        continue
                    if 300 <= status < 400 and resp.headers.get("location"):
                        return (
                            FetchResult(url, status, ctype, None, None, attempt),
                            resp.headers["location"],
                        )
                    if status >= 400:
                        return (
                            FetchResult(url, status, ctype, None, f"HTTP {status}", attempt),
                            None,
                        )
                    if ctype not in ("text/html", "application/xhtml+xml"):
                        return FetchResult(url, status, ctype, None, "not html", attempt), None
                    body = bytearray()
                    for chunk in resp.iter_bytes():
                        body += chunk
                        if len(body) > self.max_bytes:
                            return FetchResult(url, status, ctype, None, "too large", attempt), None
                    text = body.decode(resp.encoding or "utf-8", errors="replace")
                    return FetchResult(url, status, ctype, text, None, attempt), None
            except httpx.TransportError as e:
                if attempt < self.max_retries:
                    self.sleep(self._backoff(attempt, None))
                    attempt += 1
                    continue
                return FetchResult(url, 0, "", None, type(e).__name__, attempt), None
