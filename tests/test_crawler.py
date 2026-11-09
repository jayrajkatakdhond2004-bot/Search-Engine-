"""Crawler tests against a local HTTP test site (no internet needed)."""

import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from minisearch.crawler import CrawlConfig, Crawler, Frontier, extract, normalize_url

PAGES = {
    "/": """<html><head><title>Home</title></head><body>
        <nav>NAVIGATION BOILERPLATE</nav>
        <main><h1>Welcome</h1><p>home page text</p>
        <a href="/a">A</a> <a href="/a#section">A again</a> <a href="/a?utm_source=x">A tracked</a>
        <a href="b">B</a> <a href="/private/secret">secret</a> <a href="/dup1">d1</a> <a href="/dup2">d2</a>
        <a href="/file.pdf">pdf</a> <a href="/redirect-out">out</a> <a href="/flaky">flaky</a>
        <a href="/missing">missing</a> <a href="https://external.example/x">external</a>
        <a href="mailto:x@y.z">mail</a> <a href="/nofollowed" rel="nofollow">nf</a></main>
        <footer>FOOTER BOILERPLATE</footer></body></html>""",
    "/a": "<html><head><title>Page A</title></head><body><main>alpha content <a href='/c'>c</a></main></body></html>",
    "/b": "<html><head><title>B</title><meta name='robots' content='noindex'></head><body>hidden <a href='/d'>d</a></body></html>",
    "/c": "<html><head><title>C</title></head><body>gamma content</body></html>",
    "/d": "<html><head><title>D</title></head><body>delta via noindex page</body></html>",
    "/dup1": "<html><head><title>Dup</title></head><body>same   content here</body></html>",
    "/dup2": "<html><head><title>Dup</title></head><body>same content here</body></html>",
    "/private/secret": "<html><body>must never be fetched</body></html>",
    "/nofollowed": "<html><body>not linked by a followable link</body></html>",
}


class Site:
    def __init__(self, robots: str | int = "User-agent: *\nDisallow: /private\n"):
        self.requests: list[tuple[float, str]] = []
        self.flaky_failures = 2
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                site.requests.append((time.monotonic(), self.path))
                path = self.path
                if path == "/robots.txt":
                    if isinstance(robots, int):
                        return self._send(robots, "text/plain", "")
                    return self._send(200, "text/plain", robots)
                if path == "/flaky":
                    if site.flaky_failures > 0:
                        site.flaky_failures -= 1
                        return self._send(503, "text/html", "busy")
                    return self._send(
                        200, "text/html", "<title>Flaky</title><body>recovered</body>"
                    )
                if path == "/file.pdf":
                    return self._send(200, "application/pdf", "%PDF-1.4")
                if path == "/redirect-out":
                    self.send_response(302)
                    self.send_header("Location", "http://offsite.invalid/")
                    self.end_headers()
                    return None
                if path in PAGES:
                    return self._send(200, "text/html; charset=utf-8", PAGES[path])
                return self._send(404, "text/html", "not found")

            def _send(self, code, ctype, body):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def paths(self):
        return [p for _, p in self.requests]

    def close(self):
        self.server.shutdown()


@pytest.fixture
def site():
    s = Site()
    yield s
    s.close()


def make_crawler(site, tmp_path, **kw):
    cfg = CrawlConfig(
        seeds=[site.url],
        delay=kw.pop("delay", 0.0),
        state_dir=tmp_path / "state",
        backoff_base=0.01,
        **kw,
    )
    client = httpx.Client(
        trust_env=False, follow_redirects=True, timeout=5, headers={"User-Agent": cfg.user_agent}
    )
    return Crawler(cfg, client=client, sleep=kw.get("sleep", time.sleep))


def read_docs(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_full_crawl_behaviour(site, tmp_path):
    out = tmp_path / "crawl.jsonl"
    crawler = make_crawler(site, tmp_path)
    crawler.run(out)
    docs = read_docs(out)
    by_path = {d["url"].replace(site.url[:-1], ""): d for d in docs}

    assert set(by_path) == {"/", "/a", "/c", "/d", "/dup1", "/flaky"}
    paths = Counter(site.paths())
    assert "/private/secret" not in paths  # robots.txt respected
    assert "/nofollowed" not in paths  # rel=nofollow respected
    assert paths["/a"] == 1  # #fragment and utm_ variants deduplicated
    assert paths["/robots.txt"] == 1  # cached per host
    assert paths["/flaky"] == 3  # two 503s, then success

    s = crawler.stats
    assert s.duplicates == 1  # /dup2 has the same text as /dup1
    assert s.noindex == 1 and "/b" not in by_path  # noindex not written, but its links followed
    assert s.skipped_non_html == 1 and s.offsite_redirects == 1
    assert s.retries == 2 and s.errors_by_kind == {"HTTP 404": 1}
    assert not any("external.example" in u for u in crawler.frontier.seen)

    home = by_path["/"]
    assert home["title"] == "Home" and "home page text" in home["body"]
    assert "BOILERPLATE" not in home["body"]  # nav/footer stripped
    assert by_path["/c"]["depth"] == 2


def test_rate_limit_per_host(site, tmp_path):
    crawler = make_crawler(site, tmp_path, delay=0.15, max_pages=4)
    crawler.run(tmp_path / "out.jsonl")
    times = [t for t, p in site.requests if p != "/robots.txt" and p != "/flaky"]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert len(gaps) >= 3
    assert min(gaps) >= 0.14, gaps


def test_crawl_delay_from_robots_is_honoured(tmp_path):
    s = Site(robots="User-agent: *\nCrawl-delay: 0.2\n")
    try:
        crawler = make_crawler(s, tmp_path, delay=0.0, max_pages=3)
        crawler.run(tmp_path / "out.jsonl")
        times = [t for t, p in s.requests if p not in ("/robots.txt", "/flaky")]
        assert min(b - a for a, b in zip(times, times[1:], strict=False)) >= 0.19
    finally:
        s.close()


def test_resume_from_saved_frontier(site, tmp_path):
    out = tmp_path / "crawl.jsonl"
    first = make_crawler(site, tmp_path, max_pages=2)
    assert first.run(out) == 2
    assert (tmp_path / "state" / "frontier.json").exists()

    second = make_crawler(site, tmp_path, max_pages=100)  # same state dir: resumes
    second.run(out)
    docs = read_docs(out)
    assert len(docs) == 6 and len({d["url"] for d in docs}) == 6
    page_fetches = Counter(p for p in site.paths() if p not in ("/robots.txt", "/flaky"))
    assert max(page_fetches.values()) == 1  # nothing fetched twice across runs


def test_depth_limit(site, tmp_path):
    crawler = make_crawler(site, tmp_path, max_depth=1)
    crawler.run(tmp_path / "out.jsonl")
    assert "/a" in site.paths() and "/c" not in site.paths()


@pytest.mark.parametrize("status,allowed", [(404, True), (500, False)])
def test_robots_error_semantics(tmp_path, status, allowed):
    s = Site(robots=status)
    try:
        crawler = make_crawler(s, tmp_path, max_pages=1)
        crawler.run(tmp_path / "out.jsonl")
        assert (crawler.stats.written == 1) is allowed
        assert crawler.stats.robots_blocked == (0 if allowed else 1)
    finally:
        s.close()


def test_unreachable_host_is_an_error_not_a_crash(tmp_path):
    cfg = CrawlConfig(
        seeds=["http://127.0.0.1:9/"], state_dir=tmp_path / "s", max_retries=1, backoff_base=0.01
    )
    client = httpx.Client(trust_env=False, timeout=1)
    c = Crawler(cfg, client=client, sleep=lambda s: None)
    c.robots._cache["127.0.0.1:9"] = True  # skip robots, test the fetch path
    assert c.run(tmp_path / "out.jsonl") == 0
    assert c.stats.errors == 1 and c.stats.retries == 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HTTP://Example.COM:80/a/b?b=2&a=1#frag", "http://example.com/a/b?a=1&b=2"),
        ("https://example.com:443", "https://example.com/"),
        ("https://example.com:8443/x", "https://example.com:8443/x"),
        ("https://example.com/?utm_source=x&id=3&fbclid=y", "https://example.com/?id=3"),
        ("mailto:a@b.c", None),
        ("javascript:alert(1)", None),
        ("ftp://example.com/file", None),
    ],
)
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


def test_normalize_relative():
    assert normalize_url("../c?x=1", "https://ex.com/a/b/page.html") == "https://ex.com/a/c?x=1"


def test_extract_prefers_main_and_handles_missing_title():
    html = "<body><header>HDR</header><article><h1>Heading</h1><p>Body text</p></article><script>x()</script></body>"
    page = extract(html, "https://ex.com/")
    assert page.title == "Heading"
    assert page.text == "Heading\nBody text"


def test_frontier_politeness_ordering():
    f = Frontier()
    for u in ["http://a.com/1", "http://a.com/2", "http://b.com/1"]:
        f.add(u, 0)
    assert not f.add("http://a.com/1", 0)  # dedup
    url, _, wait = f.pop(now=0.0)
    assert url == "http://a.com/1" and wait == 0
    f.done(url, delay=10.0, now=0.0)
    url2, _, wait2 = f.pop(now=1.0)  # a.com is cooling down: b.com goes first
    assert url2 == "http://b.com/1" and wait2 == 0
    f.done(url2, delay=10.0, now=1.0)
    url3, _, wait3 = f.pop(now=2.0)
    assert url3 == "http://a.com/2" and wait3 == pytest.approx(8.0)


def test_frontier_release_keeps_host_scheduled():
    f = Frontier()
    f.add("http://a.com/blocked", 0)
    f.add("http://a.com/next", 0)
    url, _, _ = f.pop(now=0.0)
    f.release(url)  # no fetch happened (robots disallowed): the host must stay schedulable
    assert f.pop(now=0.0)[0] == "http://a.com/next"


def test_parse_crawl_delay_groups_and_floats():
    from minisearch.crawler.robots import parse_crawl_delay

    txt = "User-agent: *\nCrawl-delay: 2.5\n\nUser-agent: minisearch-bot\nUser-agent: other\nCrawl-delay: 0.5\n"
    assert parse_crawl_delay(txt, "minisearch-bot/0.1 (+x)") == 0.5
    assert parse_crawl_delay(txt, "SomeBot/1.0") == 2.5
    assert parse_crawl_delay("User-agent: *\nDisallow: /x\n", "a") is None
