"""IR evaluation metrics, written from their textbook definitions.

Notation: ``ranking`` is the system's ranked list of doc IDs; ``qrels`` maps
doc ID to graded relevance (> 0 means relevant).

* P@k        = |relevant in top k| / k
* R@k        = |relevant in top k| / |relevant|
* AP         = (1/|relevant|) * sum over ranks i of a relevant doc of P@i.
               Relevant docs never retrieved add 0, so recall is part of AP.
* MAP        = mean of AP over queries
* DCG@k      = sum over i=1..k of (2^rel_i - 1) / log2(i + 1)
* nDCG@k     = DCG@k / DCG@k of the ideal ordering (graded gains)
* RR         = 1 / rank of the first relevant doc

Queries without any relevant documents are skipped when averaging, the
same convention as trec_eval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def precision_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    return sum(1 for d in ranking[:k] if qrels.get(d, 0) > 0) / k


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    n_rel = sum(1 for g in qrels.values() if g > 0)
    if n_rel == 0:
        return 0.0
    return sum(1 for d in ranking[:k] if qrels.get(d, 0) > 0) / n_rel


def average_precision(ranking: Sequence[str], qrels: Mapping[str, int]) -> float:
    n_rel = sum(1 for g in qrels.values() if g > 0)
    if n_rel == 0:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, 1):
        if qrels.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / n_rel


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    gains = [max(qrels.get(d, 0), 0) for d in ranking[:k]]
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranking: Sequence[str], qrels: Mapping[str, int]) -> float:
    for i, d in enumerate(ranking, 1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


@dataclass
class EvalResult:
    p_at_10: float
    r_at_10: float
    map: float
    ndcg_at_10: float
    mrr: float
    num_queries: int

    def as_dict(self) -> dict:
        return {
            "P@10": self.p_at_10,
            "R@10": self.r_at_10,
            "MAP": self.map,
            "nDCG@10": self.ndcg_at_10,
            "MRR": self.mrr,
            "queries": self.num_queries,
        }


def evaluate_run(
    run: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, int]]
) -> EvalResult:
    """Average the metrics over every query that has at least one relevant doc."""
    qids = [q for q, rel in qrels.items() if any(g > 0 for g in rel.values())]
    if not qids:
        raise ValueError("no judged queries")
    sums = dict.fromkeys(("p", "r", "ap", "ndcg", "rr"), 0.0)
    for q in qids:
        ranking, rel = run.get(q, []), qrels[q]
        sums["p"] += precision_at_k(ranking, rel, 10)
        sums["r"] += recall_at_k(ranking, rel, 10)
        sums["ap"] += average_precision(ranking, rel)
        sums["ndcg"] += ndcg_at_k(ranking, rel, 10)
        sums["rr"] += reciprocal_rank(ranking, rel)
    n = len(qids)
    return EvalResult(
        sums["p"] / n, sums["r"] / n, sums["ap"] / n, sums["ndcg"] / n, sums["rr"] / n, n
    )


def per_query(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    metric: str = "nDCG@10",
) -> dict[str, float]:
    """Per-query scores for significance testing (judged queries only)."""
    fns = {
        "nDCG@10": lambda r, q: ndcg_at_k(r, q, 10),
        "AP": average_precision,
        "P@10": lambda r, q: precision_at_k(r, q, 10),
    }
    fn = fns[metric]
    return {
        q: fn(run.get(q, []), rel) for q, rel in qrels.items() if any(g > 0 for g in rel.values())
    }


def paired_randomization_test(
    a: Mapping[str, float], b: Mapping[str, float], iterations: int = 20_000, seed: int = 0
) -> float:
    """Two-sided p-value for H0 "systems A and B are equally good" (Fisher's sign-flip test).

    Under H0 the labels A and B are exchangeable per query, so each per-query
    difference flips sign with probability 1/2. The p-value is the fraction of
    random sign assignments whose |mean difference| is at least the observed one.
    """
    import numpy as np

    qs = sorted(set(a) & set(b))
    diffs = np.array([a[q] - b[q] for q in qs])
    observed = abs(diffs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(iterations, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (iterations + 1))
