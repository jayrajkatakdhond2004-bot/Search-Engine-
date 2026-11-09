"""Performance benchmarks on Simple English Wikipedia.

    python benchmarks/run_benchmarks.py --input data/simplewiki.jsonl --limit 100000

Measures

1. Indexing throughput: docs/s and tokens/s for analysis, inversion,
   segment flushes and background merges, then the final force-merge.
   Documents are pre-loaded into memory so JSON parsing is not counted.
2. Index size on disk: the same postings written raw (fixed uint32, absolute
   values) and compressed (d-gaps + VByte), per file, plus the source text size.
3. Query latency over 1,000 queries built from *held-out* article titles
   (titles of articles that were not indexed), so queries look like real
   searches and cannot be tuned to the index:
   600 free-text, 150 AND, 150 phrase, 100 prefix-wildcard.
   p50/p95/p99 per type, BM25 and TF-IDF, after a warm-up pass. Timings
   include fetching the top-10 stored documents (not snippets).
4. Engines on the free-text queries: vectorised TAAT vs DAAT exhaustive vs
   DAAT+WAND (latency and number of documents fully scored).
5. Memory: peak RSS while indexing, RSS of the opened index, RSS after queries.

Outputs: benchmarks/results/benchmark.{md,json}, latency.csv and three charts in docs/images/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import random
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import charts  # noqa: E402
from minisearch import Index  # noqa: E402
from minisearch.datasets.wikipedia import read_jsonl  # noqa: E402
from minisearch.indexer.merge import merge_segments  # noqa: E402
from minisearch.query.parser import escape  # noqa: E402
from minisearch.ranking import BM25, TfIdf  # noqa: E402
from minisearch.storage.disk_segment import DiskSegment  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"
IMAGES = ROOT / "docs" / "images"
MB = 1024 * 1024


class PeakRSS:
    """Samples this process's RSS every 50 ms in a background thread."""

    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / MB


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


# --------------------------------------------------------------------- indexing


def bench_indexing(docs: list[dict], path: Path) -> dict:
    if path.exists():
        shutil.rmtree(path)
    gc.collect()
    base_rss = rss_mb()
    with PeakRSS() as mem:
        t0 = time.perf_counter()
        idx = Index(path, max_buffered_docs=10_000)
        for d in docs:
            idx.add(d)
        idx.commit()
        t_index = time.perf_counter() - t0
        segments_before = len(idx.manifest.segments)
        t1 = time.perf_counter()
        idx.force_merge()
        t_merge = time.perf_counter() - t1
    stats = idx.stats()
    idx.close()
    return {
        "docs": len(docs),
        "tokens": stats["total_tokens"],
        "index_seconds": round(t_index, 2),
        "force_merge_seconds": round(t_merge, 2),
        "docs_per_sec": round(len(docs) / t_index, 1),
        "tokens_per_sec": round(stats["total_tokens"] / t_index),
        "segments_before_force_merge": segments_before,
        "peak_rss_mb": round(mem.peak / MB, 1),
        "rss_growth_mb": round(mem.peak / MB - base_rss, 1),
        "terms": stats["terms"],
    }


def bench_sizes(path: Path, docs: list[dict]) -> dict:
    idx = Index(path)
    (meta,) = idx.manifest.segments
    seg = DiskSegment(idx._files[meta.name], 0, frozenset())
    raw_dir = path.parent / "raw_copy"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir()
    raw_meta = merge_segments([seg], raw_dir, "seg_raw", "raw")  # same postings, uncompressed
    text_bytes = sum(len(d.get("title", "")) + len(d.get("body", "")) for d in docs)
    out = {
        "source_text_mb": round(text_bytes / MB, 1),
        "compressed": {k: round(v / MB, 2) for k, v in meta.file_sizes.items()},
        "raw": {k: round(v / MB, 2) for k, v in raw_meta.file_sizes.items()},
    }
    for name, sizes in (("compressed", meta.file_sizes), ("raw", raw_meta.file_sizes)):
        out[f"{name}_postings_mb"] = round((sizes["post"] + sizes["pos"]) / MB, 2)
        out[f"{name}_index_mb"] = round((sizes["post"] + sizes["pos"] + sizes["dict"]) / MB, 2)
    out["postings_ratio"] = round(out["raw_postings_mb"] / out["compressed_postings_mb"], 2)
    out["doc_ids_tfs_ratio"] = round(raw_meta.file_sizes["post"] / meta.file_sizes["post"], 2)
    out["positions_ratio"] = round(raw_meta.file_sizes["pos"] / meta.file_sizes["pos"], 2)
    shutil.rmtree(raw_dir)
    return out


# ---------------------------------------------------------------------- queries


def build_queries(held_out: list[dict], n: int = 1000, seed: int = 7) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    titles = [escape(d["title"]) for d in held_out if escape(d["title"])]
    rng.shuffle(titles)
    multi = [t for t in titles if len(t.split()) >= 2]
    words = [w for t in titles for w in t.split() if len(w) >= 5 and w.isalpha()]
    qs: list[tuple[str, str]] = []
    qs += [("free-text", t) for t in titles[:600]]
    for t in multi[:150]:
        a, b = rng.sample(t.split(), 2)
        qs.append(("AND", f"{a} AND {b}"))
    qs += [("phrase", f'"{t}"') for t in multi[150:300]]
    qs += [("wildcard", rng.choice(words)[: rng.randint(4, 5)] + "*") for _ in range(100)]
    assert len(qs) == n, len(qs)
    return qs


def bench_latency(idx: Index, queries: list[tuple[str, str]]) -> dict:
    s = idx.searcher()
    for _, q in queries[:100]:  # warm-up: page cache, dictionary blocks
        s.search(q, k=10)
    out: dict = {"rows": []}
    for rname, ranker in (("bm25", BM25()), ("tfidf", TfIdf())):
        lat: dict[str, list[float]] = {}
        hits: list[int] = []
        for qtype, q in queries:
            t0 = time.perf_counter()
            res = s.search(q, k=10, ranker=ranker)
            ms = (time.perf_counter() - t0) * 1000
            lat.setdefault(qtype, []).append(ms)
            lat.setdefault("all", []).append(ms)
            hits.append(res.total_hits)
            if rname == "bm25":
                out["rows"].append(
                    {"type": qtype, "query": q, "ms": round(ms, 3), "hits": res.total_hits}
                )
        out[rname] = {
            t: {
                "n": len(v),
                "mean": round(statistics.mean(v), 2),
                "p50": round(pct(v, 50), 2),
                "p95": round(pct(v, 95), 2),
                "p99": round(pct(v, 99), 2),
                "max": round(max(v), 2),
            }
            for t, v in lat.items()
        }
        out[rname]["qps_single_thread"] = round(1000 / statistics.mean(lat["all"]), 1)
        out[rname]["median_hits"] = int(statistics.median(hits))
    return out


def bench_engines(idx: Index, queries: list[tuple[str, str]], n: int = 300) -> dict:
    s = idx.searcher()
    free = [q for t, q in queries if t == "free-text"][:n]
    configs = {
        "vector (numpy TAAT)": dict(engine="vector"),
        "DAAT exhaustive": dict(engine="daat", use_wand=False),
        "DAAT + WAND": dict(engine="daat", use_wand=True),
    }
    out = {}
    for name, kw in configs.items():
        lat, scored, skipped = [], [], []
        for q in free:
            t0 = time.perf_counter()
            res = s.search(q, k=10, load_fields=False, **kw)
            lat.append((time.perf_counter() - t0) * 1000)
            st = res.stats
            scored.append(st.get("docs_scored", st.get("postings_scored", 0)))
            skipped.append(st.get("blocks_skipped", 0))
        out[name] = {
            "p50": round(pct(lat, 50), 2),
            "p95": round(pct(lat, 95), 2),
            "p99": round(pct(lat, 99), 2),
            "mean": round(statistics.mean(lat), 2),
            "mean_scored": round(statistics.mean(scored), 1),
            "mean_blocks_skipped": round(statistics.mean(skipped), 1),
        }
    # sanity: identical top-10 rankings
    mismatches = 0
    for q in free[:100]:
        a = [h.doc_id for h in s.search(q, k=10, engine="vector", load_fields=False).hits]
        b = [h.doc_id for h in s.search(q, k=10, engine="daat", load_fields=False).hits]
        mismatches += a != b
    out["ranking_mismatches_vs_vector_in_100"] = mismatches
    return out


_MEM_PROBE = """
import json, sys, time, psutil
sys.path.insert(0, sys.argv[3])
proc = psutil.Process()
before = proc.memory_info().rss
from minisearch import Index
imported = proc.memory_info().rss
t0 = time.perf_counter()
idx = Index(sys.argv[1])
s = idx.searcher()
opened = proc.memory_info().rss
open_ms = (time.perf_counter() - t0) * 1000
for q in json.load(open(sys.argv[2])):
    s.search(q, k=10)
after = proc.memory_info().rss
mb = 1024 * 1024
print(json.dumps({"interpreter_mb": round(before / mb, 1), "libraries_mb": round(imported / mb, 1),
                  "opened_mb": round(opened / mb, 1), "open_ms": round(open_ms, 1),
                  "after_queries_mb": round(after / mb, 1)}))
"""


def bench_query_memory(path: Path, queries: list[tuple[str, str]]) -> dict:
    """Query-time memory measured in a *fresh* process.

    Measuring in the indexing process would be misleading: CPython's allocator
    does not return freed memory to the OS, so RSS stays inflated by indexing.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([q for _, q in queries], f)
    out = subprocess.run(
        [sys.executable, "-c", _MEM_PROBE, str(path), f.name, str(ROOT / "src")],
        capture_output=True,
        text=True,
        check=True,
    )
    Path(f.name).unlink()
    return json.loads(out.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------- outputs


def write_outputs(r: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    rows = r["latency"].pop("rows")
    with open(RESULTS / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["type", "query", "ms", "hits"])
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "benchmark.json").write_text(json.dumps(r, indent=1))

    ix, sz, lat, eng, mem = r["indexing"], r["sizes"], r["latency"], r["engines"], r["memory"]
    md = [
        "# Performance benchmark",
        "",
        f"Simple English Wikipedia, **{ix['docs']:,} articles** ({sz['source_text_mb']} MB of text, "
        f"{ix['tokens']:,} indexed tokens, {ix['terms']:,} distinct terms). "
        f"Machine: {r['machine']}. Single process, single thread.",
        "",
        "## Indexing",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Indexing time (analysis + inversion + flushes + merges) | {ix['index_seconds']} s |",
        f"| Throughput | **{ix['docs_per_sec']:,.0f} docs/s** · {ix['tokens_per_sec']:,} tokens/s |",
        f"| Segments before force-merge | {ix['segments_before_force_merge']} |",
        f"| Force-merge to 1 segment | {ix['force_merge_seconds']} s |",
        f"| Peak RSS while indexing | {ix['peak_rss_mb']} MB |",
        "",
        "## Index size on disk",
        "",
        "| File | Raw (uint32) | Delta + VByte | Ratio |",
        "|---|---:|---:|---:|",
        f"| Postings: doc IDs + tfs (`.post`) | {sz['raw']['post']} MB | {sz['compressed']['post']} MB | {sz['doc_ids_tfs_ratio']}× |",
        f"| Positions (`.pos`) | {sz['raw']['pos']} MB | {sz['compressed']['pos']} MB | {sz['positions_ratio']}× |",
        f"| **Postings total** | **{sz['raw_postings_mb']} MB** | **{sz['compressed_postings_mb']} MB** | **{sz['postings_ratio']}×** |",
        f"| Term dictionary (`.dict`, front-coded) | {sz['raw']['dict']} MB | {sz['compressed']['dict']} MB | |",
        f"| Inverted index total | {sz['raw_index_mb']} MB | {sz['compressed_index_mb']} MB | |",
        f"| Doc store (zlib JSON, for snippets) | | {sz['compressed']['docs']} MB | |",
        "",
        f"The compressed inverted index is **{sz['compressed_index_mb'] / sz['source_text_mb'] * 100:.0f}% "
        f"of the source text size**, positions included.",
        "",
        "## Query latency (1,000 queries, top-10, warm)",
        "",
        "| Query type | n | BM25 p50 | p95 | p99 | TF-IDF p50 | p95 | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in ("free-text", "AND", "phrase", "wildcard", "all"):
        b, f_ = lat["bm25"][t], lat["tfidf"][t]
        name = "**all**" if t == "all" else t
        md.append(
            f"| {name} | {b['n']} | {b['p50']} ms | {b['p95']} ms | {b['p99']} ms | "
            f"{f_['p50']} ms | {f_['p95']} ms | {f_['p99']} ms |"
        )
    md += [
        "",
        f"Single-thread throughput: {lat['bm25']['qps_single_thread']} queries/s (BM25). "
        f"Median hits per query: {lat['bm25']['median_hits']:,}.",
        "",
        "## Engines: vectorised TAAT vs DAAT vs WAND (300 free-text queries)",
        "",
        "| Engine | p50 | p95 | p99 | work per query |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("vector (numpy TAAT)", "DAAT exhaustive", "DAAT + WAND"):
        e = eng[name]
        unit = "postings scored" if name.startswith("vector") else "docs fully scored"
        md.append(
            f"| {name} | {e['p50']} ms | {e['p95']} ms | {e['p99']} ms | {e['mean_scored']:,.0f} {unit} |"
        )
    md += [
        "",
        f"Top-10 rankings identical across engines: {100 - eng['ranking_mismatches_vs_vector_in_100']}/100 checked.",
        "",
        "## Memory",
        "",
        "| Phase | RSS |",
        "|---|---:|",
        f"| Indexing process: peak while indexing {ix['docs']:,} docs "
        f"(includes the {ix['docs']:,} source docs held in memory) | {ix['peak_rss_mb']} MB |",
        f"| Fresh query process: Python + numpy + minisearch imported | {mem['libraries_mb']} MB |",
        f"| … index opened, searcher ready ({mem['open_ms']} ms) | {mem['opened_mb']} MB |",
        f"| … after the 1,000-query workload | {mem['after_queries_mb']} MB |",
        "",
        "Opening is cheap because postings, positions and the doc store are memory-mapped. Only "
        "the sparse term index (1 of every 16 terms), the external id list and the doc-length "
        "array are loaded. Pages of the mapped files that queries touch count towards RSS, but "
        "they are file-backed and the OS can reclaim them under memory pressure.",
        "",
    ]
    (RESULTS / "benchmark.md").write_text("\n".join(md))

    # Chart 1: latency by query type (p50/p95/p99), BM25
    fig, ax = charts.setup(8, 4.2)
    types = ["free-text", "AND", "phrase", "wildcard"]
    series = {p: [lat["bm25"][t][p] for t in types] for p in ("p50", "p95", "p99")}
    charts.grouped_bars(ax, types, series, fmt="{:.1f}")
    ax.set_title(f"Query latency by type, BM25, {ix['docs'] // 1000}k Wikipedia articles")
    ax.set_ylabel("milliseconds (lower is better)")
    ax.set_ylim(0, max(series["p99"]) * 1.18)
    charts.save(fig, IMAGES / "latency.png")

    # Chart 2: index size, raw vs compressed
    fig, ax = charts.setup(8, 4.2)
    parts = ["Doc IDs + tfs", "Positions", "Dictionary"]
    series = {
        "raw uint32": [sz["raw"]["post"], sz["raw"]["pos"], sz["raw"]["dict"]],
        "delta + VByte": [
            sz["compressed"]["post"],
            sz["compressed"]["pos"],
            sz["compressed"]["dict"],
        ],
    }
    charts.grouped_bars(ax, parts, series, fmt="{:.1f} MB")
    ax.set_title(f"Index size on disk: postings compress {sz['postings_ratio']}×")
    ax.set_ylabel("MB (lower is better)")
    ax.set_ylim(0, max(series["raw uint32"]) * 1.18)
    charts.save(fig, IMAGES / "index_size.png")

    # Chart 3: engines
    fig, ax = charts.setup(8, 4.2)
    names = ["vector (numpy TAAT)", "DAAT exhaustive", "DAAT + WAND"]
    series = {p: [eng[n][p] for n in names] for p in ("p50", "p95")}
    charts.grouped_bars(ax, names, series, fmt="{:.1f}")
    ax.set_title("Engines on free-text queries: CPython favours numpy")
    ax.set_ylabel("milliseconds (lower is better)")
    ax.set_ylim(0, max(series["p95"]) * 1.18)
    charts.save(fig, IMAGES / "engines.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=str(ROOT / "data" / "simplewiki.jsonl"))
    ap.add_argument("--limit", type=int, default=100_000)
    ap.add_argument("--index", default=str(ROOT / "indexes" / "bench"))
    args = ap.parse_args()

    baseline = round(rss_mb(), 1)
    print(f"loading {args.limit:,} docs + held-out titles ...", flush=True)
    all_docs = list(read_jsonl(args.input, args.limit + 5000))
    docs, held_out = all_docs[: args.limit], all_docs[args.limit :]
    if len(held_out) < 2000:  # dataset exhausted: hold out from the tail instead
        docs, held_out = all_docs[:-5000], all_docs[-5000:]

    path = Path(args.index)
    print("indexing ...", flush=True)
    indexing = bench_indexing(docs, path)
    print(json.dumps(indexing), flush=True)
    sizes = bench_sizes(path, docs)
    print(json.dumps({k: v for k, v in sizes.items() if not isinstance(v, dict)}), flush=True)
    queries = build_queries(held_out)
    del all_docs, docs
    gc.collect()

    idx = Index(path)
    print("latency ...", flush=True)
    latency = bench_latency(idx, queries)
    print(json.dumps({k: v for k, v in latency.items() if k != "rows"}), flush=True)
    print("engines ...", flush=True)
    engines = bench_engines(idx, queries)
    print(json.dumps(engines), flush=True)
    memory = {"baseline_mb": baseline, **bench_query_memory(path, queries)}
    print(json.dumps(memory), flush=True)
    machine = f"{platform.processor() or platform.machine()}, {psutil.cpu_count()} vCPU, Python {platform.python_version()}"
    write_outputs(
        {
            "machine": machine,
            "indexing": indexing,
            "sizes": sizes,
            "latency": latency,
            "engines": engines,
            "memory": memory,
        }
    )
    print(f"wrote {RESULTS / 'benchmark.md'}")


if __name__ == "__main__":
    main()
