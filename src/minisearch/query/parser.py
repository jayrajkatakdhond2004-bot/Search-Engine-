"""Query language: lexer + recursive-descent parser producing an AST.

Syntax::

    information retrieval            free text (implicit operator, default OR)
    cats AND dogs                    conjunction (also &&)
    cats OR dogs                     disjunction (also ||)
    NOT dogs, -dogs                  negation
    "exact phrase"                   phrase (positional match)
    comput*   te?t   c*ter           wildcard (* = any run, ? = one char)
    (a OR b) AND NOT "c d"           grouping

Precedence, loosest first: ``OR`` < implicit adjacency < ``AND`` < ``NOT``.
In an implicit sequence, negated clauses are always required exclusions:
``a b -c`` means ``(a OR b) AND NOT c``.
Operators are recognised only in UPPERCASE. Lowercase "and"/"or"/"not" are
ordinary words, which the analyzer usually drops as stopwords.

Grammar::

    query    := or_expr EOF
    or_expr  := seq ( OR seq )*
    seq      := and_expr ( and_expr )*          # joined by the default operator
    and_expr := unary ( AND unary )*
    unary    := ( NOT | '-' ) unary | primary
    primary  := '(' or_expr ')' | PHRASE | WORD

Terms are *not* analysed here. The AST keeps the raw text, and
:func:`normalize` resolves it against the index's analyzer. Stopwords and
words that analyse into several tokens (``e-mail``) are handled there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from minisearch.analysis import Analyzer


class QuerySyntaxError(ValueError):
    """Raised for malformed queries (unbalanced parentheses, dangling operators)."""


# ---------------------------------------------------------------------------- AST


class Node:
    """Base class of query AST nodes."""


@dataclass(frozen=True)
class Term(Node):
    term: str  # an analysed term

    def __str__(self) -> str:
        return self.term


@dataclass(frozen=True)
class Phrase(Node):
    terms: tuple[str, ...]  # analysed terms
    offsets: tuple[int, ...]  # position of each term relative to the first

    def __str__(self) -> str:
        return '"' + " ".join(self.terms) + '"'


@dataclass(frozen=True)
class Wildcard(Node):
    pattern: str  # lowercased, '*' and '?' wildcards

    def __str__(self) -> str:
        return self.pattern

    @property
    def literal_prefix(self) -> str:
        return re.split(r"[*?]", self.pattern, maxsplit=1)[0]

    def regex(self) -> re.Pattern[str]:
        parts = (".*" if ch == "*" else "." if ch == "?" else re.escape(ch) for ch in self.pattern)
        return re.compile("".join(parts))


@dataclass(frozen=True)
class And(Node):
    children: tuple[Node, ...]

    def __str__(self) -> str:
        return "(" + " AND ".join(map(str, self.children)) + ")"


@dataclass(frozen=True)
class Or(Node):
    children: tuple[Node, ...]

    def __str__(self) -> str:
        return "(" + " OR ".join(map(str, self.children)) + ")"


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def __str__(self) -> str:
        return f"NOT {self.child}"


@dataclass(frozen=True)
class MatchAll(Node):
    """Matches every live document. Created for pure-negative queries like ``NOT spam``."""

    def __str__(self) -> str:
        return "*:*"


# Raw (unanalysed) leaves produced by the parser.
@dataclass(frozen=True)
class RawWord(Node):
    text: str


@dataclass(frozen=True)
class RawPhrase(Node):
    text: str


# -------------------------------------------------------------------------- lexer


@dataclass(frozen=True)
class Tok:
    kind: str  # WORD PHRASE AND OR NOT LPAREN RPAREN EOF
    value: str = ""
    pos: int = 0


_OPERATORS = {"AND": "AND", "&&": "AND", "OR": "OR", "||": "OR", "NOT": "NOT"}


def tokenize(query: str) -> list[Tok]:
    toks: list[Tok] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
        elif ch == "(":
            toks.append(Tok("LPAREN", ch, i))
            i += 1
        elif ch == ")":
            toks.append(Tok("RPAREN", ch, i))
            i += 1
        elif ch == '"':
            end = query.find('"', i + 1)
            if end == -1:  # lenient: an unterminated quote runs to the end
                end = n
            toks.append(Tok("PHRASE", query[i + 1 : end], i))
            i = end + 1
        elif ch == "-" and i + 1 < n and not query[i + 1].isspace():
            toks.append(Tok("NOT", "-", i))
            i += 1
        else:
            j = i
            while j < n and not query[j].isspace() and query[j] not in '()"':
                j += 1
            word = query[i:j]
            toks.append(Tok(_OPERATORS.get(word, "WORD"), word, i))
            i = j
    toks.append(Tok("EOF", "", n))
    return toks


# ------------------------------------------------------------------------- parser


@dataclass
class _Parser:
    toks: list[Tok]
    default_op: str = "OR"
    i: int = field(default=0)

    def peek(self) -> Tok:
        return self.toks[self.i]

    def take(self, kind: str | None = None) -> Tok:
        tok = self.toks[self.i]
        if kind is not None and tok.kind != kind:
            raise QuerySyntaxError(f"expected {kind} at position {tok.pos}, found {tok.kind}")
        self.i += 1
        return tok

    def parse(self) -> Node | None:
        if self.peek().kind == "EOF":
            return None
        node = self.or_expr()
        if self.peek().kind != "EOF":
            tok = self.peek()
            raise QuerySyntaxError(f"unexpected {tok.value or tok.kind!r} at position {tok.pos}")
        return node

    def or_expr(self) -> Node:
        parts = [self.seq()]
        while self.peek().kind == "OR":
            self.take()
            parts.append(self.seq())
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def seq(self) -> Node:
        parts = [self.and_expr()]
        while self.peek().kind in ("WORD", "PHRASE", "LPAREN", "NOT"):
            parts.append(self.and_expr())
        if len(parts) == 1:
            return parts[0]
        if self.default_op == "AND":
            return And(tuple(parts))
        # Implicit OR: negated clauses are *required* exclusions, as in Lucene
        # and web search. "a b -c" means (a OR b) AND NOT c, not a OR b OR NOT c.
        neg = [p for p in parts if isinstance(p, Not)]
        pos = [p for p in parts if not isinstance(p, Not)]
        if not neg:
            return Or(tuple(parts))
        if not pos:
            return And(tuple(neg))
        return And(((pos[0] if len(pos) == 1 else Or(tuple(pos))), *neg))

    def and_expr(self) -> Node:
        parts = [self.unary()]
        while self.peek().kind == "AND":
            self.take()
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def unary(self) -> Node:
        if self.peek().kind == "NOT":
            self.take()
            return Not(self.unary())
        return self.primary()

    def primary(self) -> Node:
        tok = self.peek()
        if tok.kind == "LPAREN":
            self.take()
            node = self.or_expr()
            if self.peek().kind == "RPAREN":
                self.take()
            elif self.peek().kind != "EOF":  # lenient: a missing ')' at the end is fine
                raise QuerySyntaxError(f"expected ')' at position {self.peek().pos}")
            return node
        if tok.kind == "PHRASE":
            self.take()
            return RawPhrase(tok.value)
        if tok.kind == "WORD":
            self.take()
            return RawWord(tok.value)
        raise QuerySyntaxError(f"unexpected {tok.value or tok.kind!r} at position {tok.pos}")


def parse(query: str, default_op: str = "OR") -> Node | None:
    """Parse ``query`` into a raw AST (leaves are :class:`RawWord` / :class:`RawPhrase`)."""
    default_op = default_op.upper()
    if default_op not in ("AND", "OR"):
        raise ValueError("default_op must be 'AND' or 'OR'")
    return _Parser(tokenize(query), default_op).parse()


# ------------------------------------------------------------------ normalisation


def _analyzed_leaf(text: str, analyzer: Analyzer) -> Node | None:
    analyzed = analyzer.analyze(text)
    if not analyzed:
        return None  # only stopwords / punctuation
    if len(analyzed) == 1:
        return Term(analyzed[0][0])
    first = analyzed[0][1]
    return Phrase(tuple(t for t, _ in analyzed), tuple(p - first for _, p in analyzed))


def _wildcard_text(text: str, analyzer: Analyzer) -> str:
    cfg = analyzer.config
    if cfg.lowercase:
        text = text.lower()
    if cfg.fold_accents:
        from minisearch.analysis.analyzer import _fold_accents

        text = _fold_accents(text)
    return text


def normalize(node: Node | None, analyzer: Analyzer) -> Node | None:
    """Analyse the leaves and simplify the tree.

    * words/phrases go through the index analyzer. Stopword-only leaves
      disappear, and multi-token words become phrases.
    * single-child AND/OR collapse. ``NOT NOT x`` becomes ``x``.
    * a query whose positive part is empty but has negations (``NOT spam``)
      becomes ``MatchAll AND NOT spam``.
    """
    if node is None:
        return None
    if isinstance(node, RawWord):
        text = node.text
        if ("*" in text or "?" in text) and text.strip("*?"):
            return Wildcard(_wildcard_text(text, analyzer))
        return _analyzed_leaf(text, analyzer)
    if isinstance(node, RawPhrase):
        return _analyzed_leaf(node.text, analyzer)
    if isinstance(node, Not):
        child = normalize(node.child, analyzer)
        if child is None:
            return None
        if isinstance(child, Not):
            return child.child
        return Not(child)
    if isinstance(node, (And, Or)):
        kids = [k for k in (normalize(c, analyzer) for c in node.children) if k is not None]
        if not kids:
            return None
        if isinstance(node, And):
            flat: list[Node] = []
            for k in kids:
                flat.extend(k.children if isinstance(k, And) else (k,))
            if all(isinstance(k, Not) for k in flat):
                flat.insert(0, MatchAll())
            return flat[0] if len(flat) == 1 else And(tuple(flat))
        flat = []
        for k in kids:
            flat.extend(k.children if isinstance(k, Or) else (k,))
        return flat[0] if len(flat) == 1 else Or(tuple(flat))
    return node


_PLAIN_WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*")


def escape(text: str) -> str:
    """Turn arbitrary text into a plain bag-of-words query with no operators.

    Benchmark queries (Cranfield, SciFact, Wikipedia titles) contain ``(``,
    ``-``, ``*``, quotes and uppercase AND/OR/NOT, which would otherwise be read
    as query syntax.
    """
    return " ".join(w.lower() for w in _PLAIN_WORD.findall(text))


def parse_query(query: str, analyzer: Analyzer, default_op: str = "OR") -> Node | None:
    """Parse and normalise: the AST the engines execute."""
    node = normalize(parse(query, default_op), analyzer)
    if isinstance(node, Not):
        node = And((MatchAll(), node))
    return node
