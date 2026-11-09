"""Simple English Wikipedia dump loader.

Streams ``simplewiki-latest-pages-articles.xml.bz2`` with ``iterparse``, so
memory stays flat however large the dump. Pages that are not articles
(namespace != 0) and redirects are skipped, wikitext markup is stripped to
plain text, and documents are yielded as ``{"id", "title", "url", "body"}``.

The wikitext cleaner is deliberately simple: regexes plus a small brace
matcher for nested templates. It is not a full MediaWiki parser, but the
output is clean prose, which is what the index and the snippets need.
"""

from __future__ import annotations

import bz2
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_TAGS_WITH_BODY = re.compile(
    r"<(math|gallery|timeline|score|syntaxhighlight|code|pre|imagemap)[^>]*>.*?</\1>", re.S | re.I
)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_FILE_LINK_START = re.compile(
    r"\[\[\s*(?:File|Image|Media|Category|[a-z]{2,3}(?:-[a-z]+)?)\s*:", re.I
)
_EMPTY_BRACKETS = re.compile(r"\(\s*[;,.:]?\s*\)|\[\s*\]")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:)])")
_LINK = re.compile(r"\[\[(?:[^|\[\]]*\|)?([^\[\]]*)\]\]")
_EXT_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]")
_BOLD_ITALIC = re.compile(r"'{2,}")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_LIST = re.compile(r"^[*#:;]+\s*", re.M)
_ENTITY = re.compile(r"&(nbsp|ndash|mdash|amp|lt|gt|quot);")
_ENTITIES = {"nbsp": " ", "ndash": "–", "mdash": "—", "amp": "&", "lt": "<", "gt": ">", "quot": '"'}
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def _strip_nested(text: str, open_: str, close: str) -> str:
    """Remove (possibly nested) ``open_ ... close`` spans, e.g. ``{{templates}}``."""
    out, depth, i, n = [], 0, 0, len(text)
    lo, lc = len(open_), len(close)
    start = 0
    while i < n:
        if text.startswith(open_, i):
            if depth == 0:
                out.append(text[start:i])
            depth += 1
            i += lo
        elif depth and text.startswith(close, i):
            depth -= 1
            i += lc
            if depth == 0:
                start = i
        else:
            i += 1
    if depth == 0:
        out.append(text[start:])
    return "".join(out)


def _strip_file_links(text: str) -> str:
    """Remove ``[[File:...]]``/``[[Category:...]]`` links, including captions with nested links."""
    out, pos = [], 0
    while (m := _FILE_LINK_START.search(text, pos)) is not None:
        out.append(text[pos : m.start()])
        depth, i, n = 1, m.end(), len(text)
        while i < n and depth:
            if text.startswith("[[", i):
                depth += 1
                i += 2
            elif text.startswith("]]", i):
                depth -= 1
                i += 2
            else:
                i += 1
        pos = i
    out.append(text[pos:])
    return "".join(out)


def clean_wikitext(text: str) -> str:
    """Convert MediaWiki markup to plain text."""
    text = _COMMENT.sub("", text)
    text = _REF.sub("", text)
    text = _TAGS_WITH_BODY.sub("", text)
    text = _strip_nested(text, "{{", "}}")
    text = _strip_nested(text, "{|", "|}")  # tables
    text = _strip_file_links(text)
    text = _LINK.sub(r"\1", text)
    text = _EXT_LINK.sub(r"\1", text)
    text = _TAG.sub("", text)
    text = _BOLD_ITALIC.sub("", text)
    text = _HEADING.sub(r"\1.", text)
    text = _LIST.sub("", text)
    text = _ENTITY.sub(lambda m: _ENTITIES[m.group(1)], text)
    text = _EMPTY_BRACKETS.sub("", text)  # e.g. "( ; )" left by removed pronunciation templates
    text = _WS.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def iter_wikipedia(
    dump_path: str | Path,
    limit: int | None = None,
    min_chars: int = 200,
    base_url: str = "https://simple.wikipedia.org/wiki/",
) -> Iterator[dict]:
    """Yield cleaned article documents from a MediaWiki XML dump (``.xml`` or ``.xml.bz2``)."""
    path = Path(dump_path)
    opener = bz2.open if path.suffix == ".bz2" else open
    count = 0
    with opener(path, "rb") as fh:
        ns_uri = ""
        for event, elem in ET.iterparse(fh, events=("start", "end")):
            if event == "start":
                if not ns_uri and elem.tag.endswith("mediawiki"):
                    ns_uri = elem.tag[: -len("mediawiki")]
                continue
            if elem.tag != f"{ns_uri}page":
                continue
            ns = elem.findtext(f"{ns_uri}ns")
            redirect = elem.find(f"{ns_uri}redirect")
            if ns == "0" and redirect is None:
                title = elem.findtext(f"{ns_uri}title") or ""
                page_id = elem.findtext(f"{ns_uri}id") or ""
                raw = elem.findtext(f"{ns_uri}revision/{ns_uri}text") or ""
                if not raw.lstrip().lower().startswith("#redirect"):
                    body = clean_wikitext(raw)
                    if len(body) >= min_chars:
                        yield {
                            "id": f"wiki:{page_id}",
                            "title": title,
                            "url": base_url + quote(title.replace(" ", "_")),
                            "body": body,
                        }
                        count += 1
                        if limit is not None and count >= limit:
                            return
            elem.clear()  # free the subtree: keeps memory flat


def write_jsonl(docs: Iterator[dict], out_path: str | Path) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)
