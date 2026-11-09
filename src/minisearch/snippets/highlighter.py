"""Query-biased snippets with highlighted query terms.

1. Re-tokenise the stored text with the *index analyzer*, keeping character
   offsets, so "Running" highlights for the query ``run`` and
   stopwords/stemming behave exactly as at search time.
2. Choose the window of ``window`` tokens that covers the most *distinct*
   query concepts, breaking ties by position (earlier is better: the lead
   text is usually the most readable, and repeated hits tend to be link
   lists). Using the match list with two pointers, this is O(m) for m matches.
3. Emit HTML. Every piece of document text is ``html.escape``d, and only our
   own ``<mark>`` tags are added, so stored content can never inject markup
   into the results page.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from minisearch.analysis import Analyzer, Token

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Snippet:
    html: str
    matched_terms: tuple[str, ...]


def _clean(text: str) -> str:
    return html.escape(_WS.sub(" ", text))


def _render(text: str, tokens: list[Token], marks: set[int], start: int, end: int) -> str:
    """Render ``text[start:end]`` with tokens whose index is in ``marks`` wrapped in <mark>."""
    out: list[str] = []
    cursor = start
    for i, tok in enumerate(tokens):
        if i not in marks or tok.end <= start or tok.start >= end:
            continue
        out.append(_clean(text[cursor : tok.start]))
        out.append("<mark>" + _clean(text[tok.start : tok.end]) + "</mark>")
        cursor = tok.end
    out.append(_clean(text[cursor:end]))
    return "".join(out).strip()


def best_window(match_idx: list[int], match_terms: list[str], window: int) -> tuple[int, int]:
    """Return ``(first, last)`` token indices of the best window over the matches."""
    best = (-1, 0, 0)  # (distinct, -start, total)
    best_span = (match_idx[0], match_idx[0])
    counts: Counter[str] = Counter()
    hi = 0
    for lo in range(len(match_idx)):
        while hi < len(match_idx) and match_idx[hi] - match_idx[lo] < window:
            counts[match_terms[hi]] += 1
            hi += 1
        key = (len(counts), -match_idx[lo], hi - lo)
        if key > best:
            best = key
            best_span = (match_idx[lo], match_idx[hi - 1])
        counts[match_terms[lo]] -= 1
        if counts[match_terms[lo]] == 0:
            del counts[match_terms[lo]]
    return best_span


def make_snippet(
    text: str,
    query_terms: Iterable[str],
    analyzer: Analyzer,
    window: int = 32,
    context_before: int = 6,
    max_scan_chars: int = 100_000,
    fallback_chars: int = 220,
    term_groups: dict[str, str] | None = None,
) -> Snippet:
    """Build a highlighted HTML snippet of ``text`` for the (analysed) ``query_terms``.

    ``term_groups`` maps terms to a concept key (e.g. all expansions of a
    wildcard share one), so the window favours covering distinct *concepts*.
    """
    qset = set(query_terms)
    groups = term_groups or {}
    scan = text[:max_scan_chars]
    tokens = list(analyzer.tokens(scan)) if qset else []
    match_idx = [i for i, t in enumerate(tokens) if t.term in qset]

    if not match_idx:
        cut = scan[:fallback_chars]
        if len(scan) > fallback_chars:
            cut = cut.rsplit(" ", 1)[0] + " …"
        return Snippet(_clean(cut).strip(), ())

    terms = [groups.get(tokens[i].term, tokens[i].term) for i in match_idx]
    first, last = best_window(match_idx, terms, window)
    lo = max(0, first - context_before)
    hi = min(len(tokens) - 1, max(last, lo + window - 1))
    start_char = 0 if lo == 0 else tokens[lo].start
    # If the window reaches the last token, keep the trailing text (closing punctuation).
    end_char = len(scan) if hi == len(tokens) - 1 and len(scan) == len(text) else tokens[hi].end
    marks = {i for i in match_idx if lo <= i <= hi}
    body = _render(scan, tokens, marks, start_char, end_char)
    prefix = "… " if start_char > 0 else ""
    suffix = " …" if end_char < len(text.rstrip()) else ""
    body = body.rstrip()
    matched = tuple(sorted({tokens[i].term for i in marks}))
    return Snippet(prefix + body + suffix, matched)


def highlight(text: str, query_terms: Iterable[str], analyzer: Analyzer) -> str:
    """Highlight every query-term occurrence in a short text (e.g. a title)."""
    qset = set(query_terms)
    tokens = list(analyzer.tokens(text))
    marks = {i for i, t in enumerate(tokens) if t.term in qset}
    return _render(text, tokens, marks, 0, len(text))
