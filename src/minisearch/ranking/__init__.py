from minisearch.ranking.base import CollectionStats, Ranker, TermScorer
from minisearch.ranking.bm25 import BM25, bm25_idf
from minisearch.ranking.tfidf import TfIdf, tfidf_idf

RANKERS: dict[str, type[Ranker]] = {"bm25": BM25, "tfidf": TfIdf}


def get_ranker(name: str, **params) -> Ranker:
    """Instantiate a ranker by name (e.g. ``get_ranker("bm25", k1=0.9, b=0.4)``)."""
    try:
        return RANKERS[name.lower()](**params)
    except KeyError:
        raise ValueError(f"unknown ranker {name!r}; choose from {sorted(RANKERS)}") from None


__all__ = [
    "BM25",
    "RANKERS",
    "CollectionStats",
    "Ranker",
    "TermScorer",
    "TfIdf",
    "bm25_idf",
    "get_ranker",
    "tfidf_idf",
]
