import pytest
from fastapi.testclient import TestClient

from minisearch import Index
from minisearch.api.app import create_app

DOCS = [
    {
        "id": "1",
        "title": "Search engines",
        "url": "https://a.test/1",
        "body": "A search engine ranks documents with BM25.",
    },
    {
        "id": "2",
        "title": "Cooking",
        "url": "https://a.test/2",
        "body": "Pasta recipes and cooking tips for engineers.",
    },
    {
        "id": "3",
        "title": "Inverted index",
        "url": "https://a.test/3",
        "body": "An inverted index maps terms to postings for search.",
    },
]


@pytest.fixture
def client(tmp_path):
    idx = Index(tmp_path / "idx")
    with TestClient(create_app(idx)) as c:
        r = c.post("/index", json=DOCS)
        assert r.status_code == 201 and r.json()["indexed"] == 3
        yield c


def test_search_returns_ranked_highlighted_results(client):
    r = client.get("/search", params={"q": "search"})
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 2 and d["total_is_exact"]
    assert [h["id"] for h in d["results"]] == ["1", "3"]
    assert "<mark>search</mark>" in d["results"][0]["snippet"]
    assert d["results"][0]["title_html"] == "<mark>Search</mark> engines"
    assert d["results"][0]["rank"] == 1 and d["parsed"] == "search"
    assert d["took_ms"] >= d["search_ms"] > 0


def test_ranker_toggle_and_params(client):
    bm = client.get(
        "/search", params={"q": "search index", "ranker": "bm25", "k1": 0.5, "b": 0.2}
    ).json()
    tf = client.get("/search", params={"q": "search index", "ranker": "tfidf"}).json()
    assert bm["ranker"] == {"name": "bm25", "k1": 0.5, "b": 0.2}
    assert tf["ranker"]["name"] == "tfidf"
    assert bm["results"][0]["score"] != tf["results"][0]["score"]


def test_pagination(client):
    p1 = client.get("/search", params={"q": "search OR cooking", "size": 2, "page": 1}).json()
    p2 = client.get("/search", params={"q": "search OR cooking", "size": 2, "page": 2}).json()
    assert p1["total"] == 3 and len(p1["results"]) == 2 and len(p2["results"]) == 1
    assert p2["results"][0]["rank"] == 3
    ids = [h["id"] for h in p1["results"] + p2["results"]]
    assert len(set(ids)) == 3


def test_boolean_phrase_wildcard_and_engines(client):
    q = lambda **p: client.get("/search", params=p).json()  # noqa: E731
    assert [h["id"] for h in q(q="search AND NOT inverted")["results"]] == ["1"]
    assert [h["id"] for h in q(q='"inverted index"')["results"]] == ["3"]
    assert {h["id"] for h in q(q="engin*")["results"]} == {"1", "2"}
    assert q(q="search index", op="and")["total"] == 1
    assert q(q="search", engine="daat")["engine"] == "daat-wand"


def test_validation_and_syntax_errors(client):
    assert client.get("/search", params={"q": "a AND"}).status_code == 400
    assert client.get("/search", params={"q": "x", "ranker": "pagerank"}).status_code == 422
    assert client.get("/search", params={"q": "x", "size": 1000}).status_code == 422
    assert client.get("/search").status_code == 422


def test_index_get_update_delete_and_stats(client):
    s = client.get("/stats").json()
    assert s["documents"] == 3 and s["buffered_documents"] == 3 and s["terms"] > 10
    client.post("/index", json={"id": "2", "title": "Cooking v2", "body": "now about kayaks"})
    assert client.get("/documents/2").json()["title"] == "Cooking v2"
    assert client.get("/search", params={"q": "pasta"}).json()["total"] == 0
    assert client.get("/search", params={"q": "kayak"}).json()["total"] == 1
    assert client.delete("/documents/2").status_code == 200
    assert client.delete("/documents/2").status_code == 404
    assert client.get("/documents/2").status_code == 404
    s = client.post("/commit").json()
    assert s["documents"] == 2 and s["segments"] == 1 and s["index_bytes"] > 0


def test_auto_id_and_commit_flag(client):
    r = client.post(
        "/index", params={"commit": True}, json={"title": "No id", "body": "generated id"}
    )
    doc_id = r.json()["ids"][0]
    assert len(doc_id) == 32 and r.json()["committed"]
    assert client.get(f"/documents/{doc_id}").json()["body"] == "generated id"


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "minisearch" in r.text
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
