from minisearch.analysis import Analyzer
from minisearch.datasets.wikipedia import clean_wikitext
from minisearch.snippets import best_window, highlight, make_snippet

A = Analyzer()


def test_highlights_stemmed_variants_with_original_casing():
    s = make_snippet("Running and RUNS: the run ended.", A.terms("run"), A)
    assert s.html == "<mark>Running</mark> and <mark>RUNS</mark>: the <mark>run</mark> ended."
    assert s.matched_terms == ("run",)
    # Porter maps "runners" to "runner" (step 4 needs m > 1), so it is correctly *not* highlighted.
    assert "<mark>" not in make_snippet("Runners", A.terms("run"), A).html


def test_window_prefers_more_distinct_terms():
    filler = " ".join(f"w{i}" for i in range(200))
    text = f"apple {filler} apple banana cherry {filler} apple apple apple"
    s = make_snippet(text, A.terms("apple banana cherry"), A, window=10)
    assert "<mark>banana</mark>" in s.html and "<mark>cherry</mark>" in s.html
    assert s.html.startswith("… ") and s.html.endswith(" …")


def test_best_window_ties_prefer_earlier():
    assert best_window([5, 50], ["a", "a"], 10) == (5, 5)
    assert best_window([0, 3, 100, 101, 102], ["a", "b", "a", "b", "c"], 10) == (100, 102)


def test_html_is_escaped():
    s = make_snippet('<script>alert("x")</script> search & <b>engine</b>', A.terms("search"), A)
    assert "<script>" not in s.html and "&lt;script&gt;" in s.html
    assert "<mark>search</mark> &amp; &lt;b&gt;engine&lt;/b&gt;" in s.html
    assert (
        highlight("<i>Search</i>", A.terms("search"), A) == "&lt;i&gt;<mark>Search</mark>&lt;/i&gt;"
    )


def test_no_match_falls_back_to_document_start():
    text = "word " * 100
    s = make_snippet(text, A.terms("missing"), A, fallback_chars=50)
    assert s.html.endswith("…") and len(s.html) < 60 and "<mark>" not in s.html


def test_whitespace_is_collapsed():
    s = make_snippet("search\n\n\n   engine", A.terms("engine"), A)
    assert s.html == "search <mark>engine</mark>"


def test_clean_wikitext():
    raw = """{{Infobox country|name=X|capital={{nested|y}}}}'''Paris''' is the [[capital city|capital]]
of [[France]].<ref>{{cite web|url=x}}</ref> See [https://example.org the site].
<!-- hidden -->
== History ==
* It is old.
{| class="wikitable"
| a || b
|}
[[File:Paris.jpg|thumb|A [[photo]] of Paris]]
[[Category:Cities]]"""
    out = clean_wikitext(raw)
    assert out.startswith("Paris is the capital\nof France. See the site.")
    assert "History." in out and "It is old." in out
    for junk in (
        "{{",
        "}}",
        "[[",
        "]]",
        "<ref",
        "wikitable",
        "File:",
        "Category:",
        "hidden",
        "'''",
    ):
        assert junk not in out


def test_clean_wikitext_nested_file_captions_and_empty_parens():
    raw = "[[File:Map.png|thumb|Detailed map of [[Istanbul]] in [[1450|1450s]].]] Constantinople ({{IPA|x}}; ) was a city."
    assert clean_wikitext(raw) == "Constantinople was a city."
