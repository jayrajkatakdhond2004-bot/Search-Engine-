"""Metric tests against hand-computed values."""

import math

import pytest

from minisearch.evaluation import (
    average_precision,
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from minisearch.evaluation.datasets import _parse_smart

# Relevant: a (grade 3), c (grade 1), x (grade 2, never retrieved).
QRELS = {"a": 3, "c": 1, "x": 2, "z": 0}
RANKING = ["b", "a", "c", "d", "z"]


def test_precision_and_recall():
    assert precision_at_k(RANKING, QRELS, 2) == 0.5  # b, a -> 1 relevant of 2
    assert precision_at_k(RANKING, QRELS, 10) == 0.2  # 2 relevant / 10 (k is fixed)
    assert recall_at_k(RANKING, QRELS, 3) == pytest.approx(2 / 3)  # a, c of {a, c, x}


def test_average_precision_hand_computed():
    # relevant at ranks 2 and 3: (1/2 + 2/3) / 3 relevant = 0.388888...
    assert average_precision(RANKING, QRELS) == pytest.approx((1 / 2 + 2 / 3) / 3)


def test_ndcg_hand_computed():
    # gains in ranked order: b=0, a=3, c=1 -> DCG = (2^3-1)/log2(3) + (2^1-1)/log2(4)
    dcg = 7 / math.log2(3) + 1 / 2
    # ideal order: 3, 2, 1 -> 7/log2(2) + 3/log2(3) + 1/log2(4)
    idcg = 7 + 3 / math.log2(3) + 1 / 2
    assert ndcg_at_k(RANKING, QRELS, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "x", "c"], QRELS, 10) == pytest.approx(1.0)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKING, QRELS) == 0.5
    assert reciprocal_rank(["q"], QRELS) == 0.0


def test_evaluate_run_skips_unjudged_queries_and_averages():
    qrels = {"q1": QRELS, "q2": {"m": 1}, "q3": {"n": 0}}  # q3 has no relevant docs
    run = {"q1": RANKING, "q2": ["m"]}
    r = evaluate_run(run, qrels)
    assert r.num_queries == 2
    assert r.map == pytest.approx(((1 / 2 + 2 / 3) / 3 + 1.0) / 2)
    assert r.mrr == pytest.approx((0.5 + 1.0) / 2)


def test_parse_smart_format(tmp_path):
    p = tmp_path / "x"
    p.write_text(".I 1\n.T\nA title\n.W\nline one\nline two\n.I 5\n.W\nsecond\n")
    assert _parse_smart(p) == [
        {"I": "1", "T": "A title", "W": "line one line two"},
        {"I": "5", "W": "second"},
    ]


def test_paired_randomization_test():
    from minisearch.evaluation import paired_randomization_test

    same = {str(i): 0.5 for i in range(50)}
    assert paired_randomization_test(same, same) == 1.0
    better = {str(i): 0.5 + 0.1 + 0.01 * (i % 3) for i in range(50)}
    assert paired_randomization_test(better, same) < 0.001
    noisy_a = {str(i): (i % 2) * 1.0 for i in range(40)}
    noisy_b = {str(i): ((i + 1) % 2) * 1.0 for i in range(40)}  # same mean, opposite queries
    assert paired_randomization_test(noisy_a, noisy_b) > 0.5
