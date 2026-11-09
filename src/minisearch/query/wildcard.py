"""Wildcard / prefix expansion against the sorted term dictionaries.

``comput*`` takes the literal prefix ``comput`` and binary-searches the sorted
term dictionary of each segment (a range scan over a sorted array has the
same complexity as walking a trie, and needs no extra structure). Patterns
with an inner wildcard (``c*ter``, ``te?t``) scan the prefix range and filter
it with a regex. A leading wildcard (``*ing``) must scan the whole
dictionary. A permuterm or k-gram index would fix that (see Future work).

Expansion is capped at ``max_expansions`` terms, keeping the ones with the
highest document frequency, so ``a*`` cannot blow up a query.

Wildcards match *indexed* terms, which are stemmed. ``comput*`` therefore
matches ``comput`` (computer, computing...) and ``computer`` (computerized).
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from minisearch.query.parser import Wildcard

if TYPE_CHECKING:
    from minisearch.query.searcher import IndexSearcher

MAX_EXPANSIONS = 128


def expand(
    searcher: IndexSearcher, node: Wildcard, max_expansions: int = MAX_EXPANSIONS
) -> list[str]:
    """Return the indexed terms matching ``node.pattern``, by descending df (ties by term)."""
    prefix = node.literal_prefix
    simple_prefix = node.pattern == prefix + "*"
    regex = None if simple_prefix else node.regex()
    candidates: set[str] = set()
    for seg in searcher.segments:
        terms = seg.terms_with_prefix(prefix) if prefix else seg.terms()
        if regex is None:
            candidates.update(terms)
        else:
            candidates.update(t for t in terms if regex.fullmatch(t))
    if len(candidates) <= max_expansions:
        return sorted(candidates, key=lambda t: (-searcher.doc_freq(t), t))
    best = heapq.nsmallest(max_expansions, candidates, key=lambda t: (-searcher.doc_freq(t), t))
    return best
