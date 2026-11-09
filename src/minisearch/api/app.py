"""HTTP API (FastAPI) and the static web UI.

Endpoints
---------
GET    /search?q=&page=&size=&ranker=&k1=&b=&op=&engine=   ranked search
POST   /index                                           add or replace one or many documents
DELETE /documents/{id}                                  delete a document
GET    /documents/{id}                                  fetch stored fields
POST   /commit                                          flush the buffer to a segment
GET    /stats                                           documents, terms, index size
GET    /health                                          liveness
GET    /                                                web UI

Concurrency: FastAPI runs these ``def`` (non-async) endpoints in a thread
pool. The :class:`Index` serialises writers with a lock. Searches run on
immutable point-in-time snapshots, so readers never block on or observe a
half-applied write.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from minisearch.indexer.index import Index
from minisearch.query import QuerySyntaxError
from minisearch.ranking import BM25, TfIdf
from minisearch.snippets import highlight, make_snippet

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


class DocumentIn(BaseModel):
    id: str | None = Field(None, description="External id; re-using an id replaces the document")
    title: str = ""
    url: str = ""
    body: str = Field("", description="Main text")


class SearchHit(BaseModel):
    rank: int
    id: str
    title: str
    title_html: str
    url: str
    snippet: str = Field(description="HTML; document text is escaped, matches wrapped in <mark>")
    score: float


class SearchResponse(BaseModel):
    query: str
    parsed: str
    ranker: dict
    engine: str
    page: int
    size: int
    total: int
    total_is_exact: bool
    took_ms: float
    search_ms: float
    results: list[SearchHit]


class IndexResponse(BaseModel):
    indexed: int
    ids: list[str]
    committed: bool


def create_app(index: Index | str | Path | None = None) -> FastAPI:
    """Build the app around an open :class:`Index` (or a path to open one)."""
    if index is None:
        index = os.environ.get("MINISEARCH_INDEX", "indexes/default")
    idx = index if isinstance(index, Index) else Index(index)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        idx.commit()

    app = FastAPI(
        title="minisearch",
        version="0.1.0",
        description="A search engine built from scratch: segmented compressed inverted index, "
        "BM25/TF-IDF, boolean/phrase/wildcard queries.",
        lifespan=lifespan,
    )
    app.state.index = idx

    @app.get("/search", response_model=SearchResponse)
    def search(
        q: str = Query(
            ..., max_length=1000, description='Query: terms, AND/OR/NOT, "phrases", wild*'
        ),
        page: int = Query(1, ge=1, le=1000),
        size: int = Query(10, ge=1, le=100),
        ranker: Literal["bm25", "tfidf"] = "bm25",
        k1: float = Query(1.2, ge=0.0, le=10.0),
        b: float = Query(0.75, ge=0.0, le=1.0),
        op: Literal["or", "and"] = "or",
        engine: Literal["auto", "vector", "daat"] = "auto",
    ) -> SearchResponse:
        t0 = time.perf_counter()
        r = BM25(k1=k1, b=b) if ranker == "bm25" else TfIdf()
        searcher = idx.searcher()
        try:
            res = searcher.search(
                q, k=size, offset=(page - 1) * size, ranker=r, default_op=op, engine=engine
            )
        except QuerySyntaxError as e:
            raise HTTPException(status_code=400, detail=f"query syntax error: {e}") from e
        analyzer = searcher.analyzer
        hits = []
        for i, h in enumerate(res.hits):
            f = h.fields
            title = f.get("title") or f.get("url") or f.get("id", "")
            hits.append(
                SearchHit(
                    rank=(page - 1) * size + i + 1,
                    id=f.get("id", ""),
                    title=title,
                    title_html=highlight(title, res.query_terms, analyzer),
                    url=f.get("url", ""),
                    snippet=make_snippet(
                        f.get("body", ""), res.query_terms, analyzer, term_groups=res.term_groups
                    ).html,
                    score=round(h.score, 6),
                )
            )
        return SearchResponse(
            query=q,
            parsed=res.parsed,
            ranker=r.describe(),
            engine=res.engine,
            page=page,
            size=size,
            total=res.total_hits,
            total_is_exact=res.total_is_exact,
            took_ms=round((time.perf_counter() - t0) * 1000, 3),
            search_ms=round(res.took_ms, 3),
            results=hits,
        )

    @app.post("/index", response_model=IndexResponse, status_code=201)
    def index_documents(
        docs: DocumentIn | list[DocumentIn] = Body(...),
        commit: bool = Query(False, description="Flush to disk immediately"),
    ) -> IndexResponse:
        batch = docs if isinstance(docs, list) else [docs]
        if len(batch) > 10_000:
            raise HTTPException(status_code=413, detail="at most 10,000 documents per request")
        ids = [idx.add(d.model_dump()) for d in batch]
        if commit:
            idx.commit()
        return IndexResponse(indexed=len(ids), ids=ids, committed=commit)

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str) -> dict:
        doc = idx.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return doc

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str) -> dict:
        if not idx.delete(doc_id):
            raise HTTPException(status_code=404, detail="document not found")
        return {"deleted": doc_id}

    @app.post("/commit")
    def commit() -> dict:
        idx.commit()
        return idx.stats()

    @app.get("/stats")
    def stats() -> dict:
        return idx.stats()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": len(idx)}

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")
    return app
