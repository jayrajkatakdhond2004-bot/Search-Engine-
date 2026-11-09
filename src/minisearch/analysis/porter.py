"""Porter stemmer, implemented from the original paper.

M.F. Porter, "An algorithm for suffix stripping", Program 14(3), 1980.

The implementation follows the published algorithm (it matches NLTK's
``PorterStemmer(mode=ORIGINAL_ALGORITHM)``) with one practical deviation that
the reference C implementation also makes: words of length <= 2 are returned
unchanged (otherwise "is" -> "i", "as" -> "a").

Terminology from the paper:

* A *consonant* is a letter other than a, e, i, o, u, and other than y when
  y is preceded by a consonant.
* Any word can be written ``[C](VC)^m[V]``; ``m`` is the *measure*.
* ``*v*``  - the stem contains a vowel.
* ``*d``   - the stem ends with a double consonant.
* ``*o``   - the stem ends consonant-vowel-consonant, where the final
  consonant is not w, x or y.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

_VOWELS = frozenset("aeiou")

Condition = Callable[[str], bool] | None


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def measure(stem: str) -> int:
    """Return Porter's measure ``m``: the number of VC sequences in ``stem``."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        cons = _is_consonant(stem, i)
        if cons and prev_vowel:
            m += 1
        prev_vowel = not cons
    return m


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    return (
        len(word) >= 3
        and _is_consonant(word, len(word) - 3)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 1)
        and word[-1] not in "wxy"
    )


def _m_gt0(stem: str) -> bool:
    return measure(stem) > 0


def _m_gt1(stem: str) -> bool:
    return measure(stem) > 1


def _apply_rules(word: str, rules: list[tuple[str, str, Condition]]) -> str:
    """Apply the rule with the longest matching suffix.

    Rules are listed so that the first suffix match is the longest one. If its
    condition fails, no other rule in the step is tried (as in the paper).
    """
    for suffix, replacement, condition in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if condition is None or condition(stem):
                return stem + replacement
            return word
    return word


def _step1a(word: str) -> str:
    return _apply_rules(
        word, [("sses", "ss", None), ("ies", "i", None), ("ss", "ss", None), ("s", "", None)]
    )


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        stem = word[:-3]
        return stem + "ee" if measure(stem) > 0 else word

    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if not _contains_vowel(stem):
                return word
            return _step1b_cleanup(stem)
    return word


def _step1b_cleanup(stem: str) -> str:
    if stem.endswith(("at", "bl", "iz")):
        return stem + "e"
    if _ends_double_consonant(stem) and stem[-1] not in "lsz":
        return stem[:-1]
    if measure(stem) == 1 and _ends_cvc(stem):
        return stem + "e"
    return stem


def _step1c(word: str) -> str:
    if word.endswith("y") and _contains_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


_STEP2 = [
    ("ational", "ate", _m_gt0),
    ("tional", "tion", _m_gt0),
    ("enci", "ence", _m_gt0),
    ("anci", "ance", _m_gt0),
    ("izer", "ize", _m_gt0),
    ("abli", "able", _m_gt0),
    ("alli", "al", _m_gt0),
    ("entli", "ent", _m_gt0),
    ("eli", "e", _m_gt0),
    ("ousli", "ous", _m_gt0),
    ("ization", "ize", _m_gt0),
    ("ation", "ate", _m_gt0),
    ("ator", "ate", _m_gt0),
    ("alism", "al", _m_gt0),
    ("iveness", "ive", _m_gt0),
    ("fulness", "ful", _m_gt0),
    ("ousness", "ous", _m_gt0),
    ("aliti", "al", _m_gt0),
    ("iviti", "ive", _m_gt0),
    ("biliti", "ble", _m_gt0),
]

_STEP3 = [
    ("icate", "ic", _m_gt0),
    ("ative", "", _m_gt0),
    ("alize", "al", _m_gt0),
    ("iciti", "ic", _m_gt0),
    ("ical", "ic", _m_gt0),
    ("ful", "", _m_gt0),
    ("ness", "", _m_gt0),
]

_STEP4 = [
    ("al", "", _m_gt1),
    ("ance", "", _m_gt1),
    ("ence", "", _m_gt1),
    ("er", "", _m_gt1),
    ("ic", "", _m_gt1),
    ("able", "", _m_gt1),
    ("ible", "", _m_gt1),
    ("ant", "", _m_gt1),
    ("ement", "", _m_gt1),
    ("ment", "", _m_gt1),
    ("ent", "", _m_gt1),
    ("ion", "", lambda s: measure(s) > 1 and s.endswith(("s", "t"))),
    ("ou", "", _m_gt1),
    ("ism", "", _m_gt1),
    ("ate", "", _m_gt1),
    ("iti", "", _m_gt1),
    ("ous", "", _m_gt1),
    ("ive", "", _m_gt1),
    ("ize", "", _m_gt1),
]


def _step5a(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        m = measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            return stem
    return word


def _step5b(word: str) -> str:
    if word.endswith("ll") and measure(word[:-1]) > 1:
        return word[:-1]
    return word


def stem(word: str) -> str:
    """Return the Porter stem of a lowercase ``word``."""
    if len(word) <= 2:
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _apply_rules(word, _STEP2)
    word = _apply_rules(word, _STEP3)
    word = _apply_rules(word, _STEP4)
    word = _step5a(word)
    word = _step5b(word)
    return word


class PorterStemmer:
    """Porter stemmer with a memoisation cache.

    Term frequencies are Zipfian, so a modest cache absorbs almost all calls
    during indexing. This is the single biggest analyzer speed-up.
    """

    def __init__(self, cache_size: int = 200_000) -> None:
        self._stem = lru_cache(maxsize=cache_size)(stem)

    def __call__(self, word: str) -> str:
        return self._stem(word)

    def cache_info(self):  # pragma: no cover - diagnostic helper
        return self._stem.cache_info()
