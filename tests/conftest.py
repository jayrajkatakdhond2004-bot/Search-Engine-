import random

import pytest

WORDS = [
    "search",
    "engine",
    "index",
    "query",
    "rank",
    "score",
    "token",
    "term",
    "document",
    "posting",
    "segment",
    "merge",
    "compress",
    "byte",
    "block",
    "skip",
    "phrase",
    "boolean",
    "wildcard",
    "prefix",
    "heap",
    "crawler",
    "robot",
    "python",
    "numpy",
    "vector",
    "cosine",
    "okapi",
    "length",
    "frequency",
    "inverse",
    "collection",
    "retrieval",
    "precision",
    "recall",
    "relevance",
    "judgment",
    "benchmark",
    "latency",
    "throughput",
    "memory",
    "disk",
    "file",
    "format",
    "header",
    "footer",
    "checksum",
]


def make_corpus(n: int, seed: int = 0, min_len: int = 3, max_len: int = 60) -> list[dict]:
    """Synthetic corpus with a Zipf-like word distribution (some terms get df > 128)."""
    rng = random.Random(seed)
    weights = [1 / (i + 1) for i in range(len(WORDS))]
    docs = []
    for i in range(n):
        k = rng.randint(min_len, max_len)
        body = " ".join(rng.choices(WORDS, weights=weights, k=k))
        title = " ".join(rng.choices(WORDS, k=3))
        docs.append({"id": f"d{i}", "title": title, "body": body, "url": f"http://x/{i}"})
    return docs


@pytest.fixture
def corpus():
    return make_corpus(600)
