"""HTML to (title, clean text, links), using BeautifulSoup with the lxml parser."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from minisearch.crawler.urls import normalize_url

# Boilerplate containers whose text should not be indexed.
_STRIP_TAGS = [
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
]
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n\s*\n+")


@dataclass
class Page:
    title: str
    text: str
    links: list[str] = field(default_factory=list)
    canonical: str | None = None
    noindex: bool = False
    nofollow: bool = False


def extract(html: str, url: str) -> Page:
    soup = BeautifulSoup(html, "lxml")

    robots = " ".join(
        m.get("content", "")
        for m in soup.find_all("meta", attrs={"name": re.compile("^robots$", re.I)})
    ).lower()
    canonical = None
    link = soup.find("link", rel=lambda v: v and "canonical" in v)
    if link and link.get("href"):
        canonical = normalize_url(link["href"], url)

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title and (h1 := soup.find("h1")):
        title = h1.get_text(" ", strip=True)

    links: list[str] = []
    if "nofollow" not in robots:
        for a in soup.find_all("a", href=True):
            if "nofollow" in (a.get("rel") or []):
                continue
            n = normalize_url(a["href"], url)
            if n:
                links.append(n)

    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    # Block-level elements become line breaks so words from adjacent blocks don't merge.
    text = root.get_text("\n", strip=True)
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n", text).strip()

    return Page(
        title=title,
        text=text,
        links=list(dict.fromkeys(links)),
        canonical=canonical,
        noindex="noindex" in robots,
        nofollow="nofollow" in robots,
    )
