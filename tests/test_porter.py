import re
from pathlib import Path

import pytest

from minisearch.analysis.porter import PorterStemmer, measure, stem

# Examples taken directly from Porter (1980).
PAPER_EXAMPLES = {
    # step 1a
    "caresses": "caress",
    "ponies": "poni",
    "ties": "ti",
    "caress": "caress",
    "cats": "cat",
    # step 1b
    "feed": "feed",
    "agreed": "agre",
    "plastered": "plaster",
    "bled": "bled",
    "motoring": "motor",
    "sing": "sing",
    "conflated": "conflat",
    "troubled": "troubl",
    "sized": "size",
    "hopping": "hop",
    "tanned": "tan",
    "falling": "fall",
    "hissing": "hiss",
    "fizzed": "fizz",
    "failing": "fail",
    "filing": "file",
    # step 1c
    "happy": "happi",
    "sky": "sky",
    # multi-step
    "relational": "relat",
    "conditional": "condit",
    "rational": "ration",
    "valenci": "valenc",
    "digitizer": "digit",
    "generalizations": "gener",
    "oscillators": "oscil",
    "triplicate": "triplic",
    "formative": "form",
    "hopefulness": "hope",
    "electrical": "electr",
    "revival": "reviv",
    "allowance": "allow",
    "adjustment": "adjust",
    "adoption": "adopt",
    "controlling": "control",
    "rolling": "roll",
    "probate": "probat",
    "rate": "rate",
    "cease": "ceas",
}


@pytest.mark.parametrize("word,expected", sorted(PAPER_EXAMPLES.items()))
def test_paper_examples(word, expected):
    assert stem(word) == expected


@pytest.mark.parametrize(
    "word,m",
    [
        ("tr", 0),
        ("ee", 0),
        ("tree", 0),
        ("y", 0),
        ("by", 0),
        ("trouble", 1),
        ("oats", 1),
        ("trees", 1),
        ("ivy", 1),
        ("troubles", 2),
        ("private", 2),
        ("oaten", 2),
        ("orrery", 2),
    ],
)
def test_measure_matches_paper_table(word, m):
    assert measure(word) == m


def test_short_words_untouched():
    assert stem("is") == "is"
    assert stem("as") == "as"


def test_cached_stemmer_agrees():
    s = PorterStemmer(cache_size=16)
    for w in PAPER_EXAMPLES:
        assert s(w) == stem(w)


def test_matches_nltk_original_algorithm_on_real_vocabulary():
    """Cross-check against NLTK's implementation of the original paper (oracle only)."""
    nltk_porter = pytest.importorskip("nltk.stem.porter")
    oracle = nltk_porter.PorterStemmer(mode=nltk_porter.PorterStemmer.ORIGINAL_ALGORITHM)

    words = set(PAPER_EXAMPLES)
    # Only the project's own sources: a virtualenv inside the repo (.venv/) holds
    # third-party files that are not all UTF-8.
    root = Path(__file__).resolve().parents[1]
    for folder in ("src", "tests"):
        for path in (root / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            words.update(re.findall(r"[a-z]{3,}", text))
    words.update(_GENERATED)
    mismatches = {w: (stem(w), oracle.stem(w)) for w in words if stem(w) != oracle.stem(w)}
    assert not mismatches, mismatches


# Productive English morphology, exercising every rule of steps 2-5.
_ROOTS = [
    "nation",
    "relate",
    "hope",
    "form",
    "general",
    "sense",
    "active",
    "happy",
    "control",
    "depend",
    "adjust",
    "conduct",
    "decide",
    "probe",
    "electric",
    "agree",
    "communicate",
]
_SUFFIXES = [
    "",
    "s",
    "es",
    "ed",
    "ing",
    "ly",
    "al",
    "ally",
    "ation",
    "ational",
    "ness",
    "ful",
    "fulness",
    "ive",
    "iveness",
    "ize",
    "izer",
    "ization",
    "able",
    "ability",
    "ible",
    "ment",
    "ments",
    "ence",
    "ance",
    "ism",
    "ist",
    "ity",
    "ous",
    "ously",
    "er",
    "ers",
]
_GENERATED = {r + s for r in _ROOTS for s in _SUFFIXES} | {
    r[:-1] + s for r in _ROOTS for s in _SUFFIXES
}
