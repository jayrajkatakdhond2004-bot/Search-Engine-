"""Document-at-a-time top-K: WAND for disjunctions, zig-zag intersection for conjunctions.

Top-K with a heap
-----------------
A size-K min-heap holds the best ``(score, -doc)`` seen so far. Its root is
the current *threshold* θ: a new doc enters only if it beats the root. Each
push costs O(log K), so a full run is O(n log K) instead of sorting all n
candidates.

WAND (Broder et al., 2003)
--------------------------
Every term t has an upper bound ``UB_t`` on any score it can contribute (from
its max tf and min doc length, see ``TermScorer.upper_bound``). Keep the
iterators sorted by current doc. Walk them, summing UB, until the sum first
exceeds θ: that iterator's doc is the *pivot*. No doc before the pivot can
beat θ, because even matching every term before the pivot cannot reach it.
All earlier iterators can therefore ``advance(pivot)``, through the skip
pointers, past documents that are never scored. As θ rises, more and more
of the postings are skipped.

Results equal exhaustive scoring exactly. A test checks this on random queries.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

from minisearch.query.parser import And, Node, Not, Or, Term
from minisearch.query.postings_iter import END, IterStats, TermIterator
from minisearch.ranking import Ranker, TermScorer

if TYPE_CHECKING:
    from minisearch.query.searcher import IndexSearcher


@dataclass
class _Clause:
    it: TermIterator
    scorer: TermScorer
    ub: float
    order: int  # original query position, so scores sum in a fixed order


class TopK:
    """Bounded min-heap of ``(score, -doc)``; ties broken by smaller doc ID."""

    __slots__ = ("k", "heap")

    def __init__(self, k: int) -> None:
        self.k = k
        self.heap: list[tuple[float, int]] = []

    @property
    def threshold(self) -> float:
        return self.heap[0][0] if len(self.heap) >= self.k else -1.0

    def push(self, score: float, doc: int) -> None:
        item = (score, -doc)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def results(self) -> list[tuple[int, float]]:
        return [(-nd, s) for s, nd in sorted(self.heap, key=lambda x: (-x[0], -x[1]))]


def supports(node: Node) -> str | None:
    """Return 'or' / 'and' if the DAAT engine can execute this query shape."""
    if isinstance(node, Term):
        return "or"
    if isinstance(node, Or) and all(isinstance(c, Term) for c in node.children):
        return "or"
    if isinstance(node, And):
        pos = [c for c in node.children if isinstance(c, Term)]
        neg = [c for c in node.children if isinstance(c, Not) and isinstance(c.child, Term)]
        if pos and len(pos) + len(neg) == len(node.children):
            return "and"
    return None


class DaatEngine:
    def __init__(self, searcher: IndexSearcher, ranker: Ranker) -> None:
        self.s = searcher
        self.ranker = ranker
        self.stats = IterStats()
        self.docs_scored = 0
        self._live = searcher.live_mask.tolist()
        self._dls = searcher.doc_lengths_global.tolist()

    def _clauses(self, terms: list[str]) -> list[_Clause]:
        from collections import Counter

        s, stats = self.s, self.s.stats
        out = []
        for order, (term, qtf) in enumerate(Counter(terms).items()):
            df = s.doc_freq(term)
            if df == 0:
                continue
            scorer = self.ranker.term_scorer(df, stats, qtf)
            ub = max(
                scorer.upper_bound(info.max_tf, info.min_dl)
                for seg in s.segments
                if (info := seg.term_info(term)) is not None
            )
            it = TermIterator(s.segments, term, self.stats)
            if it.doc != END:
                out.append(_Clause(it, scorer, ub, order))
        return out

    # ----------------------------------------------------------------- OR

    def disjunction(self, terms: list[str], k: int, use_wand: bool = True) -> TopK:
        clauses = self._clauses(terms)
        top = TopK(k)
        live, dls = self._live, self._dls
        while clauses:
            clauses.sort(key=lambda c: c.it.doc)
            theta = top.threshold
            if use_wand:
                acc_ub, pivot = 0.0, -1
                for idx, c in enumerate(clauses):
                    acc_ub += c.ub
                    if acc_ub * (1 + 1e-9) > theta:
                        pivot = idx
                        break
                if pivot == -1:
                    break  # no remaining doc can enter the top-K
            else:
                pivot = 0
            pdoc = clauses[pivot].it.doc
            if clauses[0].it.doc == pdoc:
                # Fully score pdoc: every clause positioned on it contributes.
                on = sorted((c for c in clauses if c.it.doc == pdoc), key=lambda c: c.order)
                if live[pdoc]:
                    dl = dls[pdoc]
                    score = 0.0
                    for c in on:
                        score += c.scorer.score(c.it.tf(), dl)
                    self.docs_scored += 1
                    top.push(score, pdoc)
                for c in on:
                    c.it.next()
            else:
                for c in clauses[:pivot]:
                    c.it.advance(pdoc)
            clauses = [c for c in clauses if c.it.doc != END]
        return top

    # ---------------------------------------------------------------- AND

    def conjunction(self, terms: list[str], excluded: list[str], k: int) -> tuple[TopK, int]:
        """Zig-zag intersection: the rarest iterator leads, the others ``advance`` to it."""
        from collections import Counter

        top = TopK(k)
        clauses = self._clauses(terms)
        if len(clauses) < len(Counter(terms)):
            return top, 0  # some required term matches nothing
        clauses.sort(key=lambda c: c.it.cost)
        negs = [TermIterator(self.s.segments, t, self.stats) for t in excluded]
        live, dls = self._live, self._dls
        lead, rest = clauses[0], clauses[1:]
        total = 0
        doc = lead.it.doc
        while doc != END:
            for c in rest:
                d = c.it.advance(doc)
                if d != doc:
                    doc = lead.it.advance(d)
                    break
            else:
                if live[doc] and not any(n.advance(doc) == doc for n in negs):
                    total += 1
                    dl = dls[doc]
                    score = 0.0
                    for c in sorted(clauses, key=lambda c: c.order):
                        score += c.scorer.score(c.it.tf(), dl)
                    self.docs_scored += 1
                    top.push(score, doc)
                doc = lead.it.next()
        return top, total

    def execute(self, node: Node, k: int, use_wand: bool = True) -> tuple[TopK, int | None]:
        """Run a supported query. Returns ``(top, exact_total or None)``."""
        shape = supports(node)
        if shape == "or":
            terms = [node.term] if isinstance(node, Term) else [c.term for c in node.children]
            top = self.disjunction(terms, k, use_wand)
            return top, (None if use_wand else self.docs_scored)
        if shape == "and":
            pos = [c.term for c in node.children if isinstance(c, Term)]
            neg = [c.child.term for c in node.children if isinstance(c, Not)]
            return self.conjunction(pos, neg, k)
        raise ValueError(f"DAAT engine does not support {node}")
