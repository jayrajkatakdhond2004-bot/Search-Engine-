"""Vectorised query engine: boolean masks for matching, term-at-a-time numpy scoring.

Matching
--------
Every AST node evaluates to a dense boolean mask over global doc IDs. AND,
OR and NOT become ``&``, ``|`` and ``~``. That costs O(max_doc / 8) bytes and
one numpy pass per operator. For collections up to a few million docs this
runs at memory bandwidth, and it makes arbitrary nesting (``a OR NOT (b c)``)
trivial and obviously correct.

Phrases
-------
Phrases are matched without a per-document loop. For each phrase term with
relative offset ``o``, build the keys ``doc << 32 | (pos - o)`` over all its
occurrences in candidate docs. A doc contains the phrase at start position
``p`` exactly when every term contributes the key ``doc << 32 | p``. The
phrase is therefore the intersection of the key sets, and the phrase
frequency is the number of surviving keys per doc.

Scoring
-------
``score(d) = sum over positive units`` where a unit is:

* a term: ``ranker(tf, df)``. Repeated terms add their query tf.
* a phrase: scored as a pseudo-term with tf = phrase frequency and df = number
  of docs containing the phrase (as in Lucene's PhraseQuery).
* a wildcard: the *maximum* over its expansions (dis-max). Summing would
  reward a doc for containing many spellings of one pattern.

Terms under a NOT never contribute to the score.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from minisearch.query.parser import And, MatchAll, Node, Not, Or, Phrase, Term, Wildcard
from minisearch.query.wildcard import expand
from minisearch.ranking import Ranker

if TYPE_CHECKING:
    from minisearch.query.searcher import IndexSearcher


@dataclass
class _Units:
    terms: Counter = field(default_factory=Counter)
    phrases: list[Phrase] = field(default_factory=list)
    wildcards: list[Wildcard] = field(default_factory=list)


def collect_units(node: Node | None, out: _Units | None = None, negated: bool = False) -> _Units:
    out = out or _Units()
    if node is None or negated:
        return out
    if isinstance(node, Term):
        out.terms[node.term] += 1
    elif isinstance(node, Phrase):
        out.phrases.append(node)
    elif isinstance(node, Wildcard):
        out.wildcards.append(node)
    elif isinstance(node, (And, Or)):
        for c in node.children:
            collect_units(c, out)
    elif isinstance(node, Not):
        collect_units(node.child, out, negated=True)
    return out


class VectorEngine:
    """Executes any AST. Returns the match mask and the score accumulator."""

    def __init__(self, searcher: IndexSearcher, ranker: Ranker) -> None:
        self.s = searcher
        self.ranker = ranker
        self._phrase_cache: dict[Phrase, tuple[np.ndarray, np.ndarray]] = {}
        self._expansions: dict[Wildcard, list[str]] = {}
        self.postings_scored = 0

    # ---------------------------------------------------------------- match

    def match(self, node: Node) -> np.ndarray:
        s = self.s
        if isinstance(node, Term):
            mask = np.zeros(s.max_doc, dtype=bool)
            for p in s.term_postings(node.term):
                mask[p.doc_ids] = True
            return mask
        if isinstance(node, Phrase):
            mask = np.zeros(s.max_doc, dtype=bool)
            docs, _ = self.phrase(node)
            mask[docs] = True
            return mask
        if isinstance(node, Wildcard):
            mask = np.zeros(s.max_doc, dtype=bool)
            for term in self.expansions(node):
                for p in s.term_postings(term):
                    mask[p.doc_ids] = True
            return mask
        if isinstance(node, And):
            # Evaluate the cheapest (positive) child first, then narrow with the rest.
            mask = None
            for child in sorted(node.children, key=self._cost):
                m = self.match(child)
                mask = m if mask is None else (mask & m)
                if not mask.any():
                    break
            return mask
        if isinstance(node, Or):
            mask = np.zeros(s.max_doc, dtype=bool)
            for child in node.children:
                mask |= self.match(child)
            return mask
        if isinstance(node, Not):
            return ~self.match(node.child)
        if isinstance(node, MatchAll):
            return s.live_mask.copy()
        raise TypeError(f"cannot execute {type(node).__name__}")

    def _cost(self, node: Node) -> float:
        if isinstance(node, Term):
            return self.s.doc_freq(node.term)
        if isinstance(node, Phrase):
            return min(self.s.doc_freq(t) for t in node.terms)
        if isinstance(node, Not):
            return float("inf")  # apply exclusions last
        if isinstance(node, MatchAll):
            return float(self.s.max_doc) + 1
        return float(self.s.max_doc)

    def expansions(self, node: Wildcard) -> list[str]:
        if node not in self._expansions:
            self._expansions[node] = expand(self.s, node)
        return self._expansions[node]

    # --------------------------------------------------------------- phrase

    def phrase(self, node: Phrase) -> tuple[np.ndarray, np.ndarray]:
        """``(docs, phrase_freqs)`` for docs containing the exact phrase."""
        if node in self._phrase_cache:
            return self._phrase_cache[node]
        s = self.s
        # Candidate docs: intersection of all terms' doc sets.
        order = sorted(range(len(node.terms)), key=lambda i: s.doc_freq(node.terms[i]))
        cand = None
        for i in order:
            m = np.zeros(s.max_doc, dtype=bool)
            for p in s.term_postings(node.terms[i]):
                m[p.doc_ids] = True
            cand = m if cand is None else cand & m
            if not cand.any():
                break
        keys: np.ndarray | None = None
        if cand is not None and cand.any():
            for i in order:
                term_keys = []
                for p in s.term_postings(node.terms[i]):
                    keep = cand[p.doc_ids]
                    if not keep.any():
                        continue
                    flat = p.flat_positions()
                    pos_keep = np.repeat(keep, p.tfs)
                    d = np.repeat(p.doc_ids, p.tfs)[pos_keep]
                    rel = flat[pos_keep] - node.offsets[i]
                    ok = rel >= 0
                    term_keys.append((d[ok] << 32) | rel[ok])
                k = np.concatenate(term_keys) if term_keys else np.empty(0, dtype=np.int64)
                keys = k if keys is None else np.intersect1d(keys, k, assume_unique=True)
                if len(keys) == 0:
                    break
        if keys is None or len(keys) == 0:
            result = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        else:
            docs, freqs = np.unique(keys >> 32, return_counts=True)
            result = (docs, freqs)
        self._phrase_cache[node] = result
        return result

    # --------------------------------------------------------------- score

    def score(self, node: Node) -> np.ndarray:
        s = self.s
        units = collect_units(node)
        acc = s.score_terms(list(units.terms.elements()), self.ranker)
        self.postings_scored += sum(s.doc_freq(t) for t in units.terms)
        stats = s.stats
        dls = s.doc_lengths_global
        for ph in units.phrases:
            docs, freqs = self.phrase(ph)
            if len(docs):
                scorer = self.ranker.term_scorer(len(docs), stats)
                acc[docs] += scorer.score_array(freqs, dls[docs])
        for wc in units.wildcards:
            best = np.zeros(s.max_doc, dtype=np.float64)
            for term in self.expansions(wc):
                scorer = self.ranker.term_scorer(s.doc_freq(term), stats)
                for p in s.term_postings(term):
                    sc = scorer.score_array(p.tfs, dls[p.doc_ids])
                    best[p.doc_ids] = np.maximum(best[p.doc_ids], sc)
                    self.postings_scored += len(p)
            acc += best
        return acc

    def execute(self, node: Node) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(matched_doc_ids, scores_accumulator)`` (live docs only)."""
        mask = self.match(node) & self.s.live_mask
        acc = self.score(node)
        return np.flatnonzero(mask), acc
