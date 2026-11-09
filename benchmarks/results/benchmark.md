# Performance benchmark

Simple English Wikipedia, **100,000 articles** (152.7 MB of text, 19,047,838 indexed tokens, 384,530 distinct terms). Machine: x86_64, 4 vCPU, Python 3.13.18. Single process, single thread.

## Indexing

| Metric | Value |
|---|---:|
| Indexing time (analysis + inversion + flushes + merges) | 121.22 s |
| Throughput | **825 docs/s** · 157,140 tokens/s |
| Segments before force-merge | 1 |
| Force-merge to 1 segment | 0.0 s |
| Peak RSS while indexing | 1044.5 MB |

## Index size on disk

| File | Raw (uint32) | Delta + VByte | Ratio |
|---|---:|---:|---:|
| Postings: doc IDs + tfs (`.post`) | 83.76 MB | 24.55 MB | 3.41× |
| Positions (`.pos`) | 72.66 MB | 28.95 MB | 2.51× |
| **Postings total** | **156.42 MB** | **53.49 MB** | **2.92×** |
| Term dictionary (`.dict`, front-coded) | 4.79 MB | 4.76 MB | |
| Inverted index total | 161.21 MB | 58.26 MB | |
| Doc store (zlib JSON, for snippets) | | 81.36 MB | |

The compressed inverted index is **38% of the source text size**, positions included.

## Query latency (1,000 queries, top-10, warm)

| Query type | n | BM25 p50 | p95 | p99 | TF-IDF p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| free-text | 600 | 0.98 ms | 2.99 ms | 5.73 ms | 0.87 ms | 2.84 ms | 5.76 ms |
| AND | 150 | 1.22 ms | 2.7 ms | 4.12 ms | 1.23 ms | 2.71 ms | 4.04 ms |
| phrase | 150 | 0.95 ms | 2.91 ms | 4.98 ms | 0.89 ms | 2.82 ms | 4.74 ms |
| wildcard | 100 | 1.83 ms | 6.34 ms | 8.78 ms | 1.62 ms | 6.15 ms | 8.39 ms |
| **all** | 1000 | 1.06 ms | 3.41 ms | 6.39 ms | 0.99 ms | 3.41 ms | 6.19 ms |

Single-thread throughput: 702.6 queries/s (BM25). Median hits per query: 115.

## Engines: vectorised TAAT vs DAAT vs WAND (300 free-text queries)

| Engine | p50 | p95 | p99 | work per query |
|---|---:|---:|---:|---:|
| vector (numpy TAAT) | 0.54 ms | 2.45 ms | 5.51 ms | 3,224 postings scored |
| DAAT exhaustive | 2.86 ms | 24.17 ms | 59.16 ms | 2,948 docs fully scored |
| DAAT + WAND | 3.07 ms | 13.38 ms | 20.05 ms | 942 docs fully scored |

Top-10 rankings identical across engines: 100/100 checked.

## Memory

| Phase | RSS |
|---|---:|
| Indexing process: peak while indexing 100,000 docs (includes the 100,000 source docs held in memory) | 1044.5 MB |
| Fresh query process: Python + numpy + minisearch imported | 32.0 MB |
| … index opened, searcher ready (57.4 ms) | 58.4 MB |
| … after the 1,000-query workload | 201.3 MB |

Opening is cheap because postings, positions and the doc store are memory-mapped. Only the sparse term index (1 of every 16 terms), the external id list and the doc-length array are loaded. Pages of the mapped files that queries touch count towards RSS, but they are file-backed and the OS can reclaim them under memory pressure.
