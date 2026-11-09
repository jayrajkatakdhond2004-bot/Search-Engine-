from minisearch.analysis import Analyzer, AnalyzerConfig


def test_tokenizer_splits_on_punctuation_and_keeps_contractions():
    a = Analyzer()
    assert a.raw_tokens("Hello, world! It's 2024; e-mail: x_y") == [
        "Hello",
        "world",
        "It's",
        "2024",
        "e",
        "mail",
        "x",
        "y",
    ]


def test_lowercase_stopwords_and_stemming():
    a = Analyzer()
    assert a.terms("The Running Dogs are RUNNING") == ["run", "dog", "run"]


def test_positions_leave_gaps_for_removed_stopwords():
    a = Analyzer()
    # "of" is a stopword; london keeps position 2 so phrase offsets stay exact.
    assert a.analyze("University of London") == [("univers", 0), ("london", 2)]


def test_possessive_and_apostrophes_are_normalised():
    a = Analyzer(AnalyzerConfig(stemmer=None))
    assert a.terms("John's book isn't O'Reilly's") == ["john", "book", "isnt", "oreilly"]


def test_accent_folding():
    a = Analyzer(AnalyzerConfig(stemmer=None))
    assert a.terms("Café naïve Zürich") == ["cafe", "naive", "zurich"]


def test_token_offsets_point_into_source_text():
    text = "Fast search engines"
    toks = list(Analyzer().tokens(text))
    assert [text[t.start : t.end] for t in toks] == ["Fast", "search", "engines"]
    assert [t.term for t in toks] == ["fast", "search", "engin"]


def test_configurable_pipeline():
    raw = Analyzer(AnalyzerConfig(lowercase=False, stopwords="none", stemmer=None))
    assert raw.terms("The Cats") == ["The", "Cats"]
    english = Analyzer(AnalyzerConfig(stopwords="english", stemmer=None))
    assert english.terms("what would you do") == []
    minlen = Analyzer(AnalyzerConfig(stopwords="none", stemmer=None, min_length=3))
    assert minlen.terms("a an ant") == ["ant"]


def test_config_round_trip():
    cfg = AnalyzerConfig(stopwords="english", stemmer=None, min_length=2)
    assert AnalyzerConfig.from_dict(cfg.to_dict()) == cfg


def test_empty_and_unicode_input():
    a = Analyzer()
    assert a.analyze("") == []
    assert a.analyze("!!! ... ???") == []
    assert a.terms("日本語 テキスト") == ["日本語", "テキスト"]
