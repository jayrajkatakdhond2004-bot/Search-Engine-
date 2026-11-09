"""Loaders for test collections with relevance judgments.

A collection is ``(docs, queries, qrels)``:

* ``docs``:    list of ``{"id", "title", "body"}``
* ``queries``: ``{qid: text}``
* ``qrels``:   ``{qid: {doc_id: graded relevance (> 0 = relevant)}}``
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Collection:
    name: str
    docs: list[dict]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def _parse_smart(path: Path) -> list[dict[str, str]]:
    """Parse the SMART format: records start at ``.I n``, fields are ``.T .A .B .W``."""
    records: list[dict[str, str]] = []
    field = None
    for line in path.read_text(encoding="latin-1").splitlines():
        m = re.match(r"^\.([IATBW])\b ?(.*)$", line)
        if m:
            tag, rest = m.groups()
            if tag == "I":
                records.append({"I": rest.strip()})
                field = None
            else:
                field = tag
                records[-1].setdefault(field, "")
                if rest:
                    records[-1][field] += rest + " "
        elif field is not None and records:
            records[-1][field] += line + " "
    return [{k: " ".join(v.split()) for k, v in r.items()} for r in records]


def load_cranfield(directory: str | Path) -> Collection:
    """Cranfield 1400: 1,400 aeronautics abstracts, 225 queries, graded judgments.

    Pitfalls handled here:

    * ``cran.qry`` numbers queries non-contiguously (001, 002, 004, 008...),
      but ``cranqrel`` numbers them 1..225 *by position*. Matching on the
      ``.I`` number silently scores most queries against another query's
      judgments, so queries are numbered by their order in the file.
    * Relevance codes: 1 = complete answer ... 4 = minimal interest. -1
      appears once per query as a placeholder and is treated as
      non-relevant. Graded gain = 5 - code (so 1 -> 4, ..., 4 -> 1).
    """
    d = Path(directory)
    docs = [
        {"id": r["I"], "title": r.get("T", ""), "body": r.get("W", "")}
        for r in _parse_smart(d / "cran.all.1400")
    ]
    queries = {str(i): r.get("W", "") for i, r in enumerate(_parse_smart(d / "cran.qry"), 1)}
    qrels: dict[str, dict[str, int]] = {}
    for line in (d / "cranqrel").read_text().split("\n"):
        parts = line.split()
        if len(parts) != 3:
            continue
        qid, doc, code = parts[0], parts[1], int(parts[2])
        gain = 5 - code if 1 <= code <= 4 else 0
        qrels.setdefault(qid, {})[doc] = gain
    return Collection("Cranfield", docs, queries, qrels)


def load_beir(directory: str | Path, split: str = "test") -> Collection:
    """A BEIR dataset directory (corpus.jsonl, queries.jsonl, qrels/{split}.tsv)."""
    d = Path(directory)
    docs = []
    with open(d / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs.append(
                {"id": str(r["_id"]), "title": r.get("title", ""), "body": r.get("text", "")}
            )
    qrels: dict[str, dict[str, int]] = {}
    with open(d / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        for qid, doc, score in reader:
            qrels.setdefault(str(qid), {})[str(doc)] = int(score)
    queries = {}
    with open(d / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r["_id"]) in qrels:
                queries[str(r["_id"])] = r["text"]
    names = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA"}
    return Collection(names.get(d.name, d.name), docs, queries, qrels)
