"""Command-line interface: ``minisearch <command>`` (or ``python -m minisearch``).

minisearch index   --index DIR --input docs.jsonl [--limit N] [--codec vbyte|raw]
minisearch search  --index DIR "query" [--ranker bm25|tfidf] [-k 10]
minisearch stats   --index DIR
minisearch merge   --index DIR [--max-segments 1]
minisearch serve   --index DIR [--host 0.0.0.0] [--port 8000]
minisearch wiki    --dump simplewiki.xml.bz2 --out data/simplewiki.jsonl [--limit N]
minisearch crawl   --seeds URL ... --out data/crawl.jsonl [--max-pages N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _progress(n: int, t0: float, every: int = 5000) -> None:
    if n % every == 0:
        dt = time.perf_counter() - t0
        print(f"  {n:,} docs  {n / dt:,.0f} docs/s", file=sys.stderr, flush=True)


def cmd_index(args: argparse.Namespace) -> None:
    from minisearch import Index
    from minisearch.datasets.wikipedia import read_jsonl

    idx = Index(args.index, codec=args.codec, max_buffered_docs=args.buffer)
    t0 = time.perf_counter()
    n = 0
    for doc in read_jsonl(args.input, args.limit):
        idx.add(doc)
        n += 1
        _progress(n, t0)
    idx.commit()
    if args.force_merge:
        idx.force_merge()
    dt = time.perf_counter() - t0
    print(f"indexed {n:,} docs in {dt:.1f}s ({n / dt:,.0f} docs/s)")
    print(json.dumps(idx.stats(), indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    from minisearch import Index, get_ranker
    from minisearch.snippets import make_snippet

    idx = Index(args.index)
    s = idx.searcher()
    res = s.search(args.query, k=args.k, ranker=get_ranker(args.ranker), default_op=args.op)
    print(f"{res.total_hits:,} hits in {res.took_ms:.2f} ms  parsed: {res.parsed}")
    for i, h in enumerate(res.hits, 1):
        snip = make_snippet(h.fields.get("body", ""), res.query_terms, s.analyzer).html
        snip = snip.replace("<mark>", "\033[1;33m").replace("</mark>", "\033[0m")
        print(
            f"\n{i:>2}. {h.fields.get('title')}  [{h.score:.4f}]\n    {h.fields.get('url')}\n    {snip}"
        )


def cmd_stats(args: argparse.Namespace) -> None:
    from minisearch import Index

    print(json.dumps(Index(args.index).stats(), indent=2))


def cmd_merge(args: argparse.Namespace) -> None:
    from minisearch import Index

    idx = Index(args.index)
    t0 = time.perf_counter()
    idx.force_merge(args.max_segments)
    print(f"merged in {time.perf_counter() - t0:.1f}s")
    print(json.dumps(idx.stats(), indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from minisearch.api.app import create_app

    uvicorn.run(create_app(args.index), host=args.host, port=args.port, log_level="info")


def cmd_wiki(args: argparse.Namespace) -> None:
    from minisearch.datasets.wikipedia import iter_wikipedia, write_jsonl

    t0 = time.perf_counter()
    n = write_jsonl(iter_wikipedia(args.dump, args.limit), args.out)
    print(f"wrote {n:,} articles to {args.out} in {time.perf_counter() - t0:.0f}s")


def cmd_crawl(args: argparse.Namespace) -> None:
    from minisearch.crawler import CrawlConfig, Crawler

    cfg = CrawlConfig(
        seeds=args.seeds,
        allowed_domains=args.domains or None,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay=args.delay,
        state_dir=Path(args.state),
        user_agent=args.user_agent,
        retry_failed=args.retry_failed,
    )
    crawler = Crawler(cfg)
    n = crawler.run(out_path=Path(args.out))
    print(json.dumps(crawler.stats.as_dict(), indent=2))
    print(f"wrote {n} pages to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="minisearch", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("index", help="index a JSONL file of {id,title,url,body}")
    s.add_argument("--index", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--codec", choices=["vbyte", "raw"], default="vbyte")
    s.add_argument("--buffer", type=int, default=10_000, help="docs buffered per segment flush")
    s.add_argument("--force-merge", action="store_true")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="run a query")
    s.add_argument("--index", required=True)
    s.add_argument("query")
    s.add_argument("--ranker", choices=["bm25", "tfidf"], default="bm25")
    s.add_argument("--op", choices=["or", "and"], default="or")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("stats", help="print index statistics")
    s.add_argument("--index", required=True)
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("merge", help="force-merge segments")
    s.add_argument("--index", required=True)
    s.add_argument("--max-segments", type=int, default=1)
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("serve", help="run the API + web UI")
    s.add_argument("--index", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("wiki", help="convert a Wikipedia XML dump to JSONL")
    s.add_argument("--dump", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int)
    s.set_defaults(func=cmd_wiki)

    s = sub.add_parser("crawl", help="politely crawl seed sites into JSONL")
    s.add_argument("--seeds", nargs="+", required=True)
    s.add_argument("--domains", nargs="*", help="allowed domains (default: the seeds' hosts)")
    s.add_argument("--out", default="data/crawl.jsonl")
    s.add_argument("--state", default="crawl_state", help="directory for the resumable frontier")
    s.add_argument("--max-pages", type=int, default=200)
    s.add_argument("--max-depth", type=int, default=5)
    s.add_argument("--delay", type=float, default=1.0, help="min seconds between requests per host")
    s.add_argument("--user-agent", default="minisearch-bot/0.1 (+educational crawler)")
    s.add_argument(
        "--retry-failed", action="store_true", help="re-queue transient failures on resume"
    )
    s.set_defaults(func=cmd_crawl)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
